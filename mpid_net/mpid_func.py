import torch
import torch.nn as nn

def make_train_step(model, loss_fn, optimizer, trainable=True):
    # Builds function that performs a step in the train loop
    def train_step(x, y, trainable):
        # Sets model to TRAIN mode
        if (trainable):
            model.train()
        else:
            model.eval()
        # Makes predictions
        y_prediction = model(x)
        
        # Computes loss
        loss = loss_fn(y_prediction, y)
        #print (y,y_prediction)
        #print (y,nn.Sigmoid()(y_prediction))

        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

def validation(model, test_loader, batch_size, device, event_nums=3200000):
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
