import spacy
import pandas as pd
import auxiliaryFunctions
import numpy as np
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
import torch.optim as optim
from transformers import BertForNextSentencePrediction, BertTokenizer, BertForSequenceClassification

def main():

    nlp = spacy.load("pt_core_news_md")
    model = BertForNextSentencePrediction.from_pretrained('DeepPavlov/bert-base-multilingual-cased-sentence')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.dropout = torch.nn.Dropout(0.5,False)


    # freezing previous bert layers to only fine-tune the classifier (last layer)
    for param in model.bert.parameters():
        param.requires_grad = False

    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/bert-base-multilingual-cased-sentence')
    
    def build_dataset(all_dialogs, predict):
        # Builds a train and a validation dataset from a list of dialogs
        # ----------
        # Parameters
        # ----------
        # all_dialogs: list of dialogs, where each dialog is a list of utterances
        # predict: boolean, if True, tailored distractors are selected, if False, they are random
        # ----------
        # Returns
        # ----------
        # train_dataset: TensorDataset, dataset used to train the model
        # val_dataset: TensorDataset, dataset used to validate the model

        print("BUILDING DATASET...")
        input_ids = []
        attention_masks = []
        labels = []
        nr_breaks = 0
        nr_no_breaks = 0

        all_sims = []
        for i in tqdm(range(len(all_dialogs))):
            dialog = all_dialogs[i]
            for k in range(len(dialog)):
                context = " ".join(dialog[:k])
                context_tokens = tokenizer.tokenize(context)
                j = 1
                while len(context_tokens) > 512:
                    context = " ".join(dialog[j:k])
                    context_tokens = tokenizer.tokenize(context)
                    j += 1
                gold_response = dialog[k]

                distractor1 = auxiliaryFunctions.getRandomResponse(all_dialogs,gold_response)

                # inside this if condition, tailored distractors are selected
		# if predict = False, random distractors are selected                
                if predict:
                    distractors = []
                    sim = auxiliaryFunctions.getSimilarity(gold_response,distractor1,nlp)
                    nr_its = 0
                    outer_break = False
                    distractors.append((distractor1,sim))
                    for m in range(200):
                        d = auxiliaryFunctions.getRandomResponse(all_dialogs,gold_response)
                        sim = auxiliaryFunctions.getSimilarity(gold_response,d,nlp)
                        distractors.append((d,sim))
                    distractors.sort(key=lambda x: x[1], reverse=True)

                    selected_distractor = None
                    for (d, sim) in distractors:
			# similarity threshold = 0.83
                        if sim <= 0.83:
                            all_sims.append(sim)
                            selected_distractor = d
                            break


                encoded_gold = tokenizer.encode_plus(
                                context,
                                text_pair=gold_response,
                                #add_special_tokens = True,
                                max_length = 512,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors='pt')

                encoded_distractor = tokenizer.encode_plus(
                                context,
                                text_pair=selected_distractor,
                                #add_special_tokens = True,
                                max_length = 512,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors='pt')

                # Add the encoded context and gold response to the list.
                input_ids.append(encoded_gold['input_ids'])
                attention_masks.append(encoded_gold['attention_mask'])
                labels.append(0)

                # Add the encoded context and distractor to the list.
                input_ids.append(encoded_distractor['input_ids'])
                attention_masks.append(encoded_distractor['attention_mask'])
                labels.append(1)

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.LongTensor(labels)

        from torch.utils.data import TensorDataset, random_split

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset





    def train(model, batch_size, learning_rate, train_dataset, val_dataset):
	# Fine-tunes a pre-trained model
        # ----------
        # Parameters
        # ----------
        # model: BertModel, pre-trained model to be fine-tuned
        # batch_size: int, size of each batch used in training
	# learning_rate: float, learning rate used
        # train_dataset: TensorDataset, dataset used to train the model
        # val_dataset: TensorDataset, dataset used to validate the model
        # ----------
        # Returns
        # ----------
        # None

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
        s = "batch_size: {}; learning_rate: {} \n".format(batch_size, learning_rate)


        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        model.train()
        n_epochs = 4
        max_size = 512

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = n_epochs * len(train_dataloader))
        for i in tqdm(range(n_epochs)):
            total_train_loss = 0

            for step, batch in tqdm(enumerate(train_dataloader)):

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()

                loss, logits = model(b_input_ids, attention_mask=b_input_mask,
                    next_sentence_label=b_labels)


                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            print("")
            s = "  Average training loss: {0:.3f} \n".format(avg_train_loss)
            print(s)


            # After the completion of each training epoch, measure our performance on
            # our validation set.

            validation(model, batch_size, val_dataset)

    

    def validation(model, batch_size, val_dataset):
	    # Validates a fine-tuned model
            # ----------
            # Parameters
            # ----------
            # model: BertModel, model to be validate
            # batch_size: int, size of each batch used in validation
            # val_dataset: TensorDataset, dataset used to validate the model
            # ----------
            # Returns
            # ----------
            # None
            validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

            print("Running Validation...")

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():

                    (loss, logits) = model(b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           next_sentence_label=b_labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += auxiliaryFunctions.flat_accuracy(logits, label_ids)


            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            s = "\n  Accuracy: {0:.3f} \n".format(avg_val_accuracy)
            print(s)


            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)


            s = "  Validation Loss: {0:.3f} \n".format(avg_val_loss)
            print(s)



    batch_sizes = [8]

    l_rates = [2e-5]
    

    all_dialogs = auxiliaryFunctions.getDialogs('movie_lines_preprocessed.csv')

    train_dataset, val_dataset = build_dataset(all_dialogs, True)
    # save dataset
    #torch.save(train_dataset, 'trainSet200sim.pt')
    #torch.save(val_dataset, 'valSet200sim.pt')

    # load dataset
    #train_dataset = torch.load('trainSet200sim.pt')
    #val_dataset = torch.load('valSet200sim.pt')

    sets = [(train_dataset, val_dataset)]


    for size in tqdm(batch_sizes):
        for lr in tqdm(l_rates):
            for (t_set, v_set) in sets:
                train(model,size,lr,t_set,v_set)
                validation(model,size)

                # reset model weights
                model = BertForNextSentencePrediction.from_pretrained('DeepPavlov/bert-base-multilingual-cased-sentence')
                model.to(device)

    # load model
    #model.load_state_dict(torch.load("modelBatches.pt"))
    # save model
    #torch.save(model.state_dict(), "modelBatches.pt")

if __name__ == '__main__':
    main()
