import torch
import torch.nn as nn
import torch.nn.functional as F


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):

        model.train()
        
        # Makes predictions
        y_prediction = model(x)
        
        score = F.sigmoid(y_prediction)

        print (score, y)
                
        # Computes loss
        loss = loss_fn(y_prediction, y)
        #print (y,y_prediction)
        #loss = loss_fn(nn.Sigmoid()(y_prediction), y)

        # Clear out gradients from the last step
        optimizer.zero_grad()
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

def make_test_step(model, test_loader, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def test_step(test_loader, train_device):
        # Sets model to TRAIN mode
        model.eval()
        tot_loss = 0
        ctr = 0
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            # Makes predictions
            x_batch = x_batch.to(train_device).view((-1,1,512,512))
            y_batch = y_batch.to(train_device)

            y_prediction = model(x_batch)
        
            # Computes loss
            loss = loss_fn(y_prediction, y_batch)
            tot_loss+=loss.item()
            #print (y,y_prediction)
            #loss = loss_fn(nn.Sigmoid()(y_prediction), y)
            ctr+=1
            if ctr == 5:
                break
        # Returns the loss
        return tot_loss / float(ctr)
    
    # Returns the function that will be called inside the train loop
    return test_step



def validation(model, test_loader, batch_size, device, event_nums):
    model.eval()
    predicted = 0.0
    total = 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        if batch_idx * batch_size > event_nums:
            break
        x_batch = x_batch.to(device).view((-1,1,512,512))
        y_prediction = nn.Sigmoid()(model(x_batch))
        y_truth = y_batch.to(device)

        #print (y_truth, y_prediction)

        ones = torch.ones(batch_size, 5).cuda()
        zeros = torch.zeros(batch_size, 5).cuda()
        y_prediction=torch.where(y_prediction >=0.5, ones, zeros)
        
        predicted += torch.sum(y_truth.eq(y_prediction).float()).cpu().numpy()
        total += batch_size * 5

    return float(predicted)/total
