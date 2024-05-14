from bmdal_reg.layers import *
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import copy
from torch.utils.data import DataLoader

def create_tabular_model(n_models, n_features, hidden_sizes=[128]*2, act='relu', n_outputs: int = 1, **config):
    layer_sizes = [n_features] + hidden_sizes + [n_outputs]
    layers = []
    for in_features, out_features in zip(layer_sizes[:-2], layer_sizes[1:-1]):
        layers.append(ParallelLinearLayer(n_models, in_features, out_features, **config))
        layers.append(get_parallel_act_layer(act))
    layers.append(ParallelLinearLayer(n_models, layer_sizes[-2], layer_sizes[-1],
                                      weight_init_mode='zero' if config.get('use_llz', False) else 'normal', **config))
    return ParallelSequential(n_models, *layers)

class Regressor(nn.Module):
    def __init__(self, flux, inputs=15,outputs=1, model_size=8, dropout=0.1,  scaler=StandardScaler,device=None, percentiles=None ):
        super().__init__()
        self.scaler = scaler
        self.flux = flux
        self.dropout = dropout
        self.model_size = model_size
        self.percentiles = percentiles
        self.loss = nn.MSELoss(reduction="mean")
        layers = [nn.Linear(inputs, 128), nn.Dropout(p=dropout), nn.ReLU()]
        for i in range(model_size-2):
            layers.append(nn.Linear(128,128))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(128,outputs))
        self.model = nn.Sequential(*layers)
        self.set_device(device)


    def set_device(self, device):
        self.device = device
        self.model.to(device)

        
    def set_scaler(self, scaler):
        self.scaler = scaler 

    def forward(self, x):
        y_hat = self.model(x.float())
        return y_hat

    def unscale(self, y):
        # get the index of the scaler that corresponds to the target
        scaler_features = self.scaler.feature_names_in_
        scaler_index = np.where(scaler_features == self.flux)[0][0]
        #if isinstance(self.scaler, StandardScaler):
        return y * self.scaler.scale_[scaler_index] + self.scaler.mean_[scaler_index]

    def loss_function(
        self, y, y_hat, train=True
    ):  # LZ : ToDo if given mode is predicted not to develop, set the outputs related to that mode to zero, and should not contribute to the loss

        loss = self.loss(y_hat, y.float())

        loss = torch.sum(loss)
        if not train:
            y_hat = torch.Tensor(self.unscale(y_hat.detach().cpu().numpy()))
            y = torch.Tensor(self.unscale(y.detach().cpu().numpy()))
            loss_unscaled = self.loss(y_hat, y.float())
        else:
            loss_unscaled = None
        return loss, loss_unscaled

    def train_step(self, dataloader, optimizer, epoch=None, disable_tqdm=False):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        for batch, (X, z, y, idx) in enumerate(
            dataloader, 0,
        ):  
            batch_size = len(X)
            # logging.debug(f"batch size recieved:{batch_size}")
            X = X.to(self.device)
            z = z.to(self.device)
            z_hat = self.forward(X.float())

            loss, _ = self.loss_function(z.unsqueeze(-1).float(), z_hat)

            # Backpropagation
            #optimizer.zero_grad()
            for param in self.model.parameters():
                param.grad = None            
            loss.backward()
            optimizer.step()

            loss = loss.item()

            losses.append(loss)

        average_loss = np.mean(losses) # / size
#        logging.debug(f"Loss: {average_loss:>7f}")

        return average_loss

    def validation_step(self, dataloader, scheduler=None):

        validation_loss = []
        with torch.no_grad():
            for X, z,y, _ in dataloader:
                X = X.to(self.device)
                z = z.to(self.device)
                z_hat = self.forward(X.float())
                #loss, unscaled_loss = self.loss_function(z.unsqueeze(-1).float(), z_hat, train = False)
                loss,_ = self.loss_function(z.unsqueeze(-1).float(), z_hat)
                validation_loss.append(loss.item())
                #validation_loss_unscaled.append(unscaled_loss.item())

        average_loss = np.mean(validation_loss) #/ size

        return average_loss

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        learning_rate: float = 5e-4,
        weight_decay: float = 1.0e-4,
        epochs: int = 10,
        patience: Union[None, int] = None,
        do_validation: bool = True,
        save_model: bool = False,
        save_dir: str = None,
        **cfg,  # type: ignore
    ) -> Tuple[List[float], List[float]]:

        train_loss, val_loss, model_reg = _fit_mlp(
            self,
            train_loader,
            valid_loader,
            learning_rate,
            weight_decay,
            epochs,
            patience,
            do_validation,
        )

        self.model = model_reg.model
        if save_model:
            torch.save(self.model.state_dict(), f"{save_dir}/regressor.h5")
        return train_loss, val_loss
    
    def predict(self, dataloader):

        size = len(dataloader.dataset)
        pred = []
        losses = []
        losses_unscaled = []
        popback = []

        for batch, (x, z, y, idx) in enumerate(dataloader):
            x = x.to(self.device)
            z = z.to(self.device)

            z_hat = self.forward(x.float())
            loss = self.loss_function(z.unsqueeze(-1).float(), z_hat, train=False)
            z_hat = z_hat.squeeze().detach().cpu().numpy()
            losses_unscaled.append(loss[1].item())
            loss = loss[0]
            z = self.unscale(z.squeeze().detach().cpu().numpy())
            z_hat = self.unscale(z_hat)
           
            losses.append(loss.item())
        

            try:
                pred.extend(z_hat)
            except:
                pred.extend([z_hat])

        average_loss = np.mean(losses)# / size
        
        pred = np.asarray(pred, dtype=object).flatten()
        #if np.any(np.isnan(losses_binned)):
        #    print(np.isnan(losses_binned))
        #    raise ValueError
        unscaled_avg_loss = np.mean(losses_unscaled)# / size
        popback = np.sum(popback)/size*100
        return pred, average_loss, unscaled_avg_loss


def _fit_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    learning_rate: float = 5.0e-4,
    weight_decay: float = 1.0e-4,
    epochs: int = 10,
    patience: Union[None, int] = None,
    do_validation: bool = True,
) -> Tuple[List[float], List[float], nn.Module]:
    """Fits a Multi Layer Perceptron model.

    Args:
        model (nn.Module): The model to be fit
        train_loader (DataLoader): The training data
        valid_loader (DataLoader): The validation data
        learning_rate (int, optional): The learning rate. Defaults to 5e-4.
        weight_decay (int, optional): The weight decay. Defaults to 1.0e-4.
        epochs (int, optional): The training epochs. Defaults to 10.
        patience (Union[None, int], optional): The patience value. Defaults to None.
        do_validation (bool, optional): Whether to do the validation loop (for testing purposes should be False). Defaults to True.

    Returns:
        List: The training loss
        List: The validation loss
        nn.Module: The trained model.
    """

    if not patience:
        patience = epochs
    best_loss = np.inf
    # instantiate optimiser
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # create scheduler

    train_loss = []
    val_loss = []
    counter = 0
    for epoch in range(epochs):

        loss = model.train_step(train_loader, opt, epoch=epoch)
        if isinstance(
            loss, tuple
        ):  # classifier also returns accuracy which we are not tracking atm
            loss = loss[0]
        train_loss.append(loss.item())

        if do_validation:
            validation_loss = model.validation_step(valid_loader)
            val_loss.append(validation_loss)
            if validation_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = validation_loss
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break
        else:
            best_model = model

    return train_loss, val_loss, best_model