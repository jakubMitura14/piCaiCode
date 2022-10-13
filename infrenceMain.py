#https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html
import model.LigtningModel

def infrence():
    checkPointPath="/path/to/checkpoint.ckpt"
    model = model.LigtningModel.Model.load_from_checkpoint("/path/to/checkpoint.ckpt")
    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    y_hat = model(x)
    