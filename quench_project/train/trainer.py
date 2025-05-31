
from utils.model_utils import reset_weights
import numpy as np
import torch

class Results:
    def __init__(self, train_losses= None, val_losses= None, val_accuracy_per_epoch=None, train_losses_per_batch=None):
        self.train_losses = train_losses
        self.val_losses =  val_losses
        self.val_accuracy_per_epoch = val_accuracy_per_epoch
        self.train_losses_per_batch = train_losses_per_batch

    def __repr__(self):
        return f"Results(train_losses={self.train_losses}, val_losses={self.val_losses}, val_accuracy_per_epoch={self.val_accuracy_per_epoch}, train_losses_per_batch={self.train_losses_per_batch})"
    
    def get_train_loss(self):
        return np.array(self.train_losses)
    
    def get_val_loss(self):
        return np.array(self.val_losses)
    
    def get_val_accuracy(self):
        return np.array(self.val_accuracy_per_epoch)
    
    def get_train_losses_per_batch(self):
        return np.array(self.train_losses_per_batch)
    


class Trainer:
    def __init__(self, model, loss_fn, optimizer,lr, num_epochs, train_loader=None, val_loader=None):

        self.model = model
        self.loss_fn = loss_fn 
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr=lr

        if optimizer == 'Adam': 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model = self.model.to(self.device)

        self.Results=Results()




class TrainerCNN3D(Trainer):
    def __init__(self, model, loss_fn, optimizer,lr, num_epochs, train_loader=None, val_loader=None):
        super().__init__(model, loss_fn, optimizer,lr, num_epochs, train_loader, val_loader)


    def run(self):

        train_losses = []
        val_losses = []
        val_accuracy_per_epoch = []
        train_losses_per_batch = []   
    
        for epoch in range(self.num_epochs):

            
        
            if epoch ==0:
                self.model.apply(reset_weights)
    
            # ---- Training ----
            self.model.train()
            
            total_train_loss = 0.0
            total_val_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(batch_x)
    
                loss = self.loss_fn(preds, batch_y)  
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() 
                train_losses_per_batch.append(loss.item())
    
            
            # ---- Evaluation ----
            self.model.eval()
            
            total = correct = 0
            TP = FP =FN = 0
    
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                    preds = self.model(batch_x)
                    loss = self.loss_fn(preds, batch_y)
                    total_val_loss += loss.item() 
    
                    probs = torch.sigmoid(preds)
                    predicted = (probs >= 0.5).float()
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
    
                    # Precision/Recall
                    TP += ((predicted == 1) & (batch_y == 1)).sum().item()
                    FP += ((predicted == 1) & (batch_y == 0)).sum().item()
                    FN += ((predicted == 0) & (batch_y == 1)).sum().item()
    
    
            
            avg_val_loss = total_val_loss / len(self.val_loader.dataset)
            avg_train_loss = total_train_loss / len(self.train_loader.dataset)
    
            val_accuracy = correct / total * 100
            val_precision = 100*(TP / (TP + FP)) if (TP + FP) > 0 else 0
            val_recall = 100*(TP / (TP + FN)) if (TP + FN) > 0 else 0
    
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | val Loss: {avg_val_loss:.4f} | val Acc: {val_accuracy:.2f}% | val Prec: {val_precision:.2f}% | val Recall: {val_recall:.2f}%" )
           
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracy_per_epoch.append(val_accuracy)
            # Store results in the Results object
        
        self.Results = Results(train_losses, val_losses, val_accuracy_per_epoch, train_losses_per_batch)
        torch.cuda.empty_cache()
        return self.Results
            


class TrainerVAE(Trainer):
    def __init__(self, model, loss_fn, optimizer,lr, num_epochs, train_loader=None, val_loader=None):
        super().__init__(model, loss_fn, optimizer,lr, num_epochs, train_loader, val_loader)

    def run(self):
    
        train_losses = []
        val_losses = []
        

        optimizer = self.optimizer


        for epoch in range(self.num_epochs):

            self.model.train()
            if epoch == 0:
                self.model.apply(reset_weights)  # Reset weights at the start of training

            total_loss = 0.0
            val_loss = 0.0

            for batch in self.train_loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                mu,logvar = self.model.encode(x)
                x_hat = self.model(x)

                loss = self.loss_fn(x, x_hat, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() 

            self.model.eval()
            with torch.no_grad():

                for batch in self.val_loader:
                    x = batch[0].to(self.device)
                    x_hat = self.model(x)
                    mu, logvar = self.model.encode(x)
                    
                    loss = self.loss_fn(x, x_hat, mu, logvar)
                    
                    val_loss += loss.item()


            avg_loss = total_loss / len(self.train_loader.dataset)
            avg_val_loss = val_loss / len(self.val_loader.dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} ")

            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)

        torch.cuda.empty_cache()

        return Results(train_losses=train_losses, val_losses=val_losses)



class TrainerConvAE(Trainer):
    def __init__(self, model, loss_fn, optimizer,lr, num_epochs, train_loader=None, val_loader=None):
        super().__init__(model, loss_fn, optimizer,lr, num_epochs, train_loader, val_loader)

    def run(self):
    
        train_losses = []
        val_losses = []
        

        optimizer = self.optimizer


        for epoch in range(self.num_epochs):

            self.model.train()
            if epoch == 0:
                self.model.apply(reset_weights)  # Reset weights at the start of training

            total_loss = 0.0
            val_loss = 0.0

            for batch in self.train_loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                x_hat = self.model(x)
                loss = self.loss_fn(x, x_hat)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() 

            self.model.eval()
            with torch.no_grad():

                for batch in self.val_loader:
                    x = batch[0].to(self.device)
                    x_hat= self.model(x)
                    loss = self.loss_fn(x, x_hat)
                    val_loss += loss.item()


            avg_loss = total_loss / len(self.train_loader.dataset)
            avg_val_loss = val_loss / len(self.val_loader.dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} ")

            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)

        torch.cuda.empty_cache()

        return Results(train_losses=train_losses, val_losses=val_losses)

