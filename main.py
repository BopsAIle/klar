import torch
import pytorch_lightning
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import time
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
import argparse


from klar.model import FusionModel
from klar.data import DInterface

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data","--data_dir", default='./data_process/PharmKG_8K/pro_data/', help="data directory")
    args.add_argument("-o","--output_model_name", default='8K_attention', help="model name")
    args.add_argument("--model_load_path", default='./Qwen3_checkpoint/PharmKG_8K', help="model path")
    args.add_argument("--KG_embedding_path", default='./KG_embedding/trained_999.pth', help="KG embedding path")
    args.add_argument("--max_path_len",type=int, default=6, help="ID sequence length")
    args.add_argument("--freeze", action="store_true", help="freeze Qwen")
    args.add_argument("-lr","--learning_rate",type=float, default=1e-5, help="use MoE")

    args.add_argument("-m","--moe_use", action="store_true", help="use MoE")
    args.add_argument("-me","--moe_expert_num",type=int, default=5, help="MoE expert num")
    args.add_argument("-mh","--moe_hidden_size",type=int, default=64, help="MoE hidden size")
    args.add_argument("-mk","--moe_k",type=int, default=4, help="how many experts to use for each batch element")

    args.add_argument("--KG_RNN_hidden_size", type=int, default=1024, help="KG RNN hidden size")
    args.add_argument("--KG_RNN_num_layers", type=int, default=2, help="KG RNN num layers")

    args.add_argument("-b","--batch_size",type=int, default=8, help="batch size")
    args.add_argument("--device", default='cpu', help="device")
    args.add_argument("--device_num",type=int, default=1, help="Device num")
    args.add_argument("--metric_name", default="Eval-hits1", help="metric name")
    args.add_argument("--data_loader_workers",type=int, default=10, help="data loader workers num")
    args.add_argument("-e","--max_epochs",type=int, default=10, help="max epochs")

    #Ablation part
    args.add_argument("-WO_KG","--WO_KG_embedding", action="store_true", help="use MoE")
    args.add_argument("-WO_Li","--WO_L_importance", action="store_true", help="use MoE")
    args.add_argument("-WO_Ll","--WO_L_load", action="store_true", help="use MoE")

    args.add_argument("-t","--test_main", action="store_true", help="test training process")
    args.add_argument("--test_model_path", default=None, help="test model path")

    args = args.parse_args()
    return args

        
def main(): 
    args = parse_args()
    print(args)
 
    mymodel = FusionModel(args)
    data = DInterface(args)
    
    exp_name = f"{args.output_model_name}_path-{args.max_path_len}"
    if(args.freeze):
        exp_name += '_FG'
        for p in mymodel.model.qwen.parameters():
            p.requires_grad = False
    if(args.moe_use):
        exp_name += f'_MoE-{args.moe_expert_num}-{args.moe_hidden_size}'
    if(args.WO_KG_embedding):
        exp_name += f'_WO-KG'
    if(args.WO_L_importance):
        exp_name += f'_WO-Li'
    if(args.WO_L_load):
        exp_name += f'_WO-Ll'

    # exp_name = f"{exp_name}_lr-{args.learning_rate}_{'_'.join(time.asctime().split()[:3])}"
    exp_name = f"{exp_name}_lr-{args.learning_rate}"
    print("Exp name is: ",exp_name)

    tb_logger = TensorBoardLogger(save_dir="PL_logs/", name=exp_name)
    early_callback = pytorch_lightning.callbacks.EarlyStopping(monitor="Eval-mrr", mode="max", patience=3)
    model_checkpoint = pytorch_lightning.callbacks.ModelCheckpoint(monitor=args.metric_name, mode="max",
        filename=exp_name + '/{epoch}-{Eval-hits10:.2f}-{Eval-hits1:.2f}',
        dirpath="output",
        save_weights_only=True,
        every_n_train_steps=100,
        save_top_k=1
    )
    callbacks = [early_callback, model_checkpoint]
    if(args.test_main):
        test_model = torch.load(args.test_model_path)
        mymodel.load_state_dict(test_model['state_dict'],strict=True)
        trainer = pytorch_lightning.Trainer(accelerator="gpu", devices=args.device_num,logger=False) 
        result = trainer.test(model=mymodel, datamodule=data)
        print(result)
    else:
        trainer = pytorch_lightning.Trainer(callbacks=[model_checkpoint],enable_checkpointing= True,
                                        max_epochs=args.max_epochs,accelerator="gpu", devices=args.device_num,val_check_interval=1.0, 
                                        logger=tb_logger) 
        torch.set_float32_matmul_precision("medium")
        trainer.fit(mymodel, data)
        result = trainer.test(mymodel, datamodule=data)
        print(model_checkpoint.best_model_path)
        trainer.save_checkpoint('output/'+exp_name+'/last_mmodel.ckpt')
    
if __name__ == '__main__':
    main()

    