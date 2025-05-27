
from utils.model_utils import reset_weights
import torch

class Results:
    def __init__(self, train_losses= None, test_losses= None, test_accuracy_per_epoch=None, train_losses_per_batch=None):
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.test_accuracy_per_epoch = test_accuracy_per_epoch
        self.train_losses_per_batch = train_losses_per_batch

    def __repr__(self):
        return f"Results(train_losses={self.train_losses}, test_losses={self.test_losses}, test_accuracy_per_epoch={self.test_accuracy_per_epoch}, train_losses_per_batch={self.train_losses_per_batch})"
    
    def get_train_loss(self):
        return self.train_losses
    def get_test_loss(self):
        return self.test_losses
    def get_test_accuracy(self):
        return self.test_accuracy_per_epoch
    def get_train_losses_per_batch(self):
        return self.train_losses_per_batch
    


class Trainer:
    def __init__(self, model, loss_fn, optimizer, num_epochs, train_loader=None, test_loader=None):

        self.model = model
        self.loss_fn = loss_fn 
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        if optimizer == 'Adam': 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


        self.Results=Results()


    def run(self):

        train_losses = []
        test_losses = []
        test_accuracy_per_epoch = []
        train_losses_per_batch = []   
    
        for epoch in range(self.num_epochs):

            
        
            if epoch ==0:
                self.model.apply(reset_weights)
    
            # ---- Training ----
            self.model.train()
            
            total_train_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
    
                loss = self.loss_fn(preds, batch_y)  
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)
                train_losses_per_batch.append(loss.item())
    
            avg_train_loss = total_train_loss / len(self.train_loader.dataset)
    
            # ---- Evaluation ----
            self.model.eval()
            total_test_loss = 0.0
            total = correct = 0
            TP = FP =FN = 0
    
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                
                    preds = self.model(batch_x)
                    loss = self.loss_fn(preds, batch_y)
                    total_test_loss += loss.item() * batch_x.size(0)
    
                    probs = torch.sigmoid(preds)
                    predicted = (probs >= 0.5).float()
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
    
                    # Precision/Recall
                    TP += ((predicted == 1) & (batch_y == 1)).sum().item()
                    FP += ((predicted == 1) & (batch_y == 0)).sum().item()
                    FN += ((predicted == 0) & (batch_y == 1)).sum().item()
    
    
            avg_test_loss = total_test_loss / len(self.test_loader.dataset)
            test_accuracy = correct / total * 100
            test_precision = 100*(TP / (TP + FP)) if (TP + FP) > 0 else 0
            test_recall = 100*(TP / (TP + FN)) if (TP + FN) > 0 else 0
    
    
    
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}% | Test Prec: {test_precision:.2f}% | Test Recall: {test_recall:.2f}%" )
           
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            test_accuracy_per_epoch.append(test_accuracy)
            # Store results in the Results object
        
        self.Results = Results(train_losses, test_losses, test_accuracy_per_epoch, train_losses_per_batch)
        return self.Results
            

          