# !/usr/bin/env python
# coding: utf-8
import argparse
import statistics
import sys
from datetime import datetime

from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.dataloader.deep_fake_dataset import *
from src.network.DeepFakeTSModel import DeepFakeTSModel
from src.utils import config
from src.utils.log import *
from src.utils.model_training_utlis_wtensorboard import train_model

debug_mode = False

print('current directory', os.getcwd())
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
print(f'cwd change to: {os.getcwd()}')

if (debug_mode):
    sys.argv = [''];
    del sys

checkpoint_attribs = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'epoch']

parser = argparse.ArgumentParser()
parser.add_argument("-ng", "--no_gpus", help="number of gpus",
                    type=int, default=1)
parser.add_argument("-cdn", "--cuda_device_no", help="cuda device no",
                    type=int, default=0)
parser.add_argument("-ws", "--window_size", help="windows size",
                    type=int, default=5)
parser.add_argument("-wst", "--window_stride", help="windows stride",
                    type=int, default=5)
parser.add_argument("-ks", "--kernel_size", help="kernel size",
                    type=int, default=3)
parser.add_argument("-bs", "--batch_size", help="batch size",
                    type=int, default=2)
parser.add_argument("-ep", "--epochs", help="epoch per validation cycle",
                    type=int, default=200)
parser.add_argument("-lr", "--learning_rate", help="learning rate",
                    type=float, default=3e-4)
parser.add_argument("-sml", "--seq_max_len", help="maximum sequence length",
                    type=int, default=200)
parser.add_argument("-rt", "--resume_training", help="resume training",
                    action="store_true", default=False)
parser.add_argument("-ftt", "--first_time_training", help="is this is first time training[True/False]",
                    action="store_true", default=False)
parser.add_argument("-sl", "--strict_load", help="partially or strictly load the saved model",
                    action="store_true", default=False)
parser.add_argument("-vt", "--validation_type", help="validation_type",
                    default='person')
parser.add_argument("-tvp", "--total_valid_persons", help="Total valid persons",
                    type=int, default=1)
parser.add_argument("-dfp", "--data_file_dir_base_path", help="data_file_dir_base_path",
                    default='/data/research_data/dfdc_train_data/')
parser.add_argument("-cout", "--cnn_out_channel", help="CNN out channel size",
                    type=int, default=16)
parser.add_argument("-fes", "--feature_embed_size", help="CNN feature embedding size",
                    type=int, default=256)
parser.add_argument("-lhs", "--lstm_hidden_size", help="LSTM hidden embedding size",
                    type=int, default=256)
parser.add_argument("-madf", "--matn_dim_feedforward", help="matn_dim_feedforward",
                    type=int, default=256)
parser.add_argument("-lld", "--lower_layer_dropout", help="lower layer dropout",
                    type=float, default=0.2)
parser.add_argument("-uld", "--upper_layer_dropout", help="upper layer dropout",
                    type=float, default=0.2)
parser.add_argument("-menh", "--module_embedding_nhead", help="module embedding multi-head attention nhead",
                    type=int, default=4)
parser.add_argument("-mmnh", "--multi_modal_nhead", help="multi-modal embeddings multi-head attention nhead",
                    type=int, default=4)
parser.add_argument("-gatnh", "--gat_mh_attn_nhead", help="gat_multi_head_attention_nhead",
                    type=int, default=1)
parser.add_argument("-enl", "--encoder_num_layers", help="LSTM encoder layer",
                    type=int, default=2)
parser.add_argument("-lstm_bi", "--lstm_bidirectional", help="LSTM bidirectional [True/False]",
                    action="store_true", default=False)
parser.add_argument("-fine_tune", "--fine_tune", help="Visual feature extractor fine tunning",
                    action="store_true", default=False)
parser.add_argument("-is_guiding", "--is_guiding", help="Guided by real data or not",
                    action="store_true", default=False)

parser.add_argument("-img_w", "--image_width", help="transform to image width",
                    type=int, default=config.image_width)
parser.add_argument("-img_h", "--image_height", help="transform to image height",
                    type=int, default=config.image_height)

parser.add_argument("-mcp", "--model_checkpoint_prefix", help="model checkpoint filename prefix",
                    default='deepfake')
parser.add_argument("-mcf", "--model_checkpoint_filename", help="model checkpoint filename",
                    default=None)
parser.add_argument("-rcf", "--resume_checkpoint_filename", help="resume checkpoint filename",
                    default=None)

parser.add_argument("-gattn_type", "--gat_attention_type", help="attention_type [concat/sum]",
                    default='sum')
parser.add_argument("-mmattn_type", "--mm_embedding_attn_merge_type",
                    help="mm_embedding_attn_merge_type [concat/sum]",
                    default='sum')
parser.add_argument("-gat_model", "--is_gat_attn_model", help="gat attention model",
                    action="store_true", default=False)
parser.add_argument("-motion_type", "--motion_type", help="motion_type",
                    default='gross')

parser.add_argument("-logf", "--log_filename", help="execution log filename",
                    default='exe_deepfake.log')
parser.add_argument("-logbd", "--log_base_dir", help="execution log base dir",
                    default='log')
parser.add_argument("-final_log", "--final_log_filename", help="Final result log filename",
                    default='final_results_deepfake.log')
parser.add_argument("-tb_wn", "--tb_writer_name", help="tensorboard writer name",
                    default='tb_runs/tb_deepfake')
parser.add_argument("-tbl", "--tb_log", help="tensorboard logging",
                    action="store_true", default=False)
parser.add_argument("-lm_archi", "--log_model_archi", help="log model",
                    action="store_true", default=False)

parser.add_argument("-ex_it", "--executed_number_it", help="total number of executed iteration",
                    type=int, default=-1)
parser.add_argument("-esp", "--early_stop_patience", help="total number of executed iteration",
                    type=int, default=50)
parser.add_argument("-cl", "--cycle_length", help="total number of executed iteration",
                    type=int, default=100)
parser.add_argument("-cm", "--cycle_mul", help="total number of executed iteration",
                    type=int, default=2)
parser.add_argument("-vpi", "--valid_person_index", help="valid person index",
                    type=int, default=0)

args = parser.parse_args()
cuda_device_no = args.cuda_device_no
no_gpus = args.no_gpus
resume_training = args.resume_training
first_time_training = args.first_time_training
strict_load = args.strict_load
batch_size = args.batch_size
lr = args.learning_rate
epochs_in_each_val_cycle = args.epochs
seq_max_len = args.seq_max_len
window_size = args.window_size
window_stride = args.window_stride
kernel_size = args.kernel_size
image_width = args.image_width
image_height = args.image_height

model_checkpoint_prefix = args.model_checkpoint_prefix
model_checkpoint_filename = args.model_checkpoint_filename
resume_checkpoint_filename = args.resume_checkpoint_filename
if (model_checkpoint_filename is not None):
    model_checkpoint_filename = model_checkpoint_filename
else:
    model_checkpoint_filename = f'{model_checkpoint_prefix}_{datetime.utcnow().timestamp()}.pth'

data_dir_base_path = args.data_file_dir_base_path
validation_type = args.validation_type
total_valid_persons = args.total_valid_persons

cnn_out_channel = args.cnn_out_channel
feature_embed_size = args.feature_embed_size
lstm_hidden_size = args.lstm_hidden_size
matn_dim_feedforward = args.matn_dim_feedforward
lower_layer_dropout = args.lower_layer_dropout
upper_layer_dropout = args.upper_layer_dropout
module_embedding_nhead = args.module_embedding_nhead
multi_modal_nhead = args.multi_modal_nhead
gat_mh_attn_nhead = args.gat_mh_attn_nhead
lstm_encoder_num_layers = args.encoder_num_layers
lstm_bidirectional = args.lstm_bidirectional
lstm_bidirectional = False
gat_attention_type = args.gat_attention_type
mm_embedding_attn_merge_type = args.mm_embedding_attn_merge_type
is_gat_attn_model = args.is_gat_attn_model
motion_type = args.motion_type
fine_tune = args.fine_tune

executed_number_it = args.executed_number_it
early_stop_patience = args.early_stop_patience
cycle_length = args.cycle_length
cycle_mul = args.cycle_mul
valid_person_index = args.valid_person_index

log_model_archi = args.log_model_archi
log_filename = args.log_filename
log_base_dir = args.log_base_dir
final_log_filename = args.final_log_filename
tb_writer_name = args.tb_writer_name
tb_log = args.tb_log
tb_writer = None
if (tb_log):
    tb_writer = SummaryWriter(tb_writer_name)

log_execution(log_base_dir, log_filename,
              f'validation type: {validation_type}, '
              f'valid_person_index:{valid_person_index}, is_guiding: {args.is_guiding} \n')
log_execution(log_base_dir, log_filename,
              f'window size: {window_size}, window_stride: {window_stride}, '
              f'seq_max_len:{seq_max_len}\n')
log_execution(log_base_dir, log_filename,
              f'early_stop_patience: {early_stop_patience}, '
              f'cycle_length:{cycle_length}, cycle_mul: {cycle_mul}\n')
log_execution(log_base_dir, log_filename,
              f'image_width: {image_width}, image_height: {image_height}\n')
log_execution(log_base_dir, log_filename,
              f'kernel size: {kernel_size}, lr:{lr}, epoch:{epochs_in_each_val_cycle}, batch_size:{batch_size}\n')
log_execution(log_base_dir, log_filename,
              f'cnn_out_channel: {cnn_out_channel}, feature_embed_size:{feature_embed_size}\n')
log_execution(log_base_dir, log_filename,
              f'lstm_hidden_size: {lstm_hidden_size}, matn_dim_feedforward:{matn_dim_feedforward}\n')
log_execution(log_base_dir, log_filename,
              f'lower_layer_dropout:{lower_layer_dropout}, upper_layer_dropout: {upper_layer_dropout}\n')
log_execution(log_base_dir, log_filename, f'module_embedding_nhead:{module_embedding_nhead}\n')
log_execution(log_base_dir, log_filename,
              f'multi_modal_nhead:{multi_modal_nhead}, '
              f'mm_embedding_attention_type:{mm_embedding_attn_merge_type}\n')
log_execution(log_base_dir, log_filename,
              f'gat_attention_type: {gat_attention_type}, is_gat_attn_model:{is_gat_attn_model}, gat_mh_attn_nhead:{gat_mh_attn_nhead}\n')
log_execution(log_base_dir, log_filename,
              f'encoder_num_layers: {lstm_encoder_num_layers}, '
              f'resume training:{resume_training}\n')
log_execution(log_base_dir, log_filename,
              f'data_file_dir_base_path:{data_dir_base_path}, modalities:{modalities}\n')
log_execution(log_base_dir, log_filename, f'modalities:{modalities}\n')
log_execution(log_base_dir, log_filename, f'executed_number_its: {executed_number_it}\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available() and cuda_device_no!=-1):
    device = torch.device(f'cuda:{cuda_device_no}')

log_execution(log_base_dir, log_filename, f'pytorch version: {torch.__version__}\n')
log_execution(log_base_dir, log_filename, f'GPU Availability: {device}, no_gpus: {no_gpus}\n')
if (device == 'cuda'):
    log_execution(log_base_dir, log_filename, f'Current cuda device: {torch.cuda.current_device()}\n\n')

rgb_transforms = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
depth_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transforms_modalities = {}
transforms_modalities[config.original_modality_tag] = rgb_transforms
transforms_modalities[config.fake_modality_tag] = rgb_transforms

# module_networks = [config.rgb_one_modality_tag, config.rgb_two_modality_tag]
module_networks = [config.rgb_one_modality_tag]
mm_module_properties = defaultdict(dict)
for modality in module_networks:
    mm_module_properties[modality]['cnn_in_channel'] = cnn_out_channel
    mm_module_properties[modality]['cnn_out_channel'] = cnn_out_channel
    mm_module_properties[modality]['kernel_size'] = kernel_size
    mm_module_properties[modality]['feature_embed_size'] = feature_embed_size
    mm_module_properties[modality]['lstm_hidden_size'] = lstm_hidden_size
    mm_module_properties[modality]['lstm_encoder_num_layers'] = lstm_encoder_num_layers
    mm_module_properties[modality]['lstm_bidirectional'] = lstm_bidirectional
    mm_module_properties[modality]['module_embedding_nhead'] = module_embedding_nhead
    mm_module_properties[modality]['dropout'] = lower_layer_dropout
    mm_module_properties[modality]['activation'] = 'relu'
    mm_module_properties[modality]['fine_tune'] = True
    mm_module_properties[modality]['feature_pooling_kernel'] = 5
    mm_module_properties[modality]['feature_pooling_stride'] = 5
    mm_module_properties[modality]['feature_pooling_type'] = 'max'
    mm_module_properties[modality]['lstm_dropout'] = 0.0

full_dataset = DeepFakeDataset(data_dir_base_path=data_dir_base_path,
                               modalities=modalities,
                               restricted_ids=None,
                               restricted_labels=None,
                               dataset_type='train', seq_max_len=seq_max_len,
                               window_size=window_size, window_stride=window_stride,
                               transforms_modalities=transforms_modalities,
                               metadata_filename='metadata.csv')

num_labels = full_dataset.num_labels
label_names = full_dataset.label_names
log_execution(log_base_dir, log_filename, f'total_activities: {num_labels}\n')
log_execution(log_base_dir, log_filename, f'total_activities: {full_dataset.label_names}\n')

validation_split = .1
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(full_dataset, batch_size=batch_size,
                              drop_last=False,
                              sampler=train_sampler,
                              collate_fn=pad_collate, num_workers=2)

valid_dataloader = DataLoader(full_dataset, batch_size=batch_size,
                              drop_last=False,
                              sampler=valid_sampler,
                              collate_fn=pad_collate, num_workers=2)

model = DeepFakeTSModel(mm_module_properties=mm_module_properties,
                        modalities=modalities,
                        modality_embedding_size=lstm_hidden_size,
                        batch_first=True,
                        window_size=window_size,
                        window_stride=window_stride,
                        multi_modal_nhead=multi_modal_nhead,
                        mm_embedding_attn_merge_type=mm_embedding_attn_merge_type,
                        dropout=upper_layer_dropout,
                        is_guiding=args.is_guiding,
                        module_networks=module_networks)
if (no_gpus > 1):
    gpu_list = list(range(torch.cuda.device_count()))
    model = nn.DataParallel(model, device_ids=gpu_list)
if (log_model_archi):
    log_execution(log_base_dir, log_filename, f'\n############ Model ############\n {str(model)}\n',
                  print_console=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_length, T_mult=cycle_mul)

log_execution(log_base_dir, log_filename,
              f'\n\n\tStart execution training \n\n')
log_execution(log_base_dir, log_filename, f'train_dataloader len: {len(train_dataloader)}\n')
log_execution(log_base_dir, log_filename, f'valid_dataloader len: {len(valid_dataloader)}\n')
log_execution(log_base_dir, log_filename,
              f'train dataset len: {len(train_dataloader.dataset)}, train dataloader len: {len(train_dataloader)}\n')
log_execution(log_base_dir, log_filename,
              f'valid dataset len: {len(valid_dataloader.dataset)}, valid dataloader len: {len(valid_dataloader)}\n')

model_save_base_dir = 'trained_model'
if (resume_checkpoint_filename is not None):
    resume_checkpoint_filepath = f'{model_save_base_dir}/{resume_checkpoint_filename}'
    if (os.path.exists(resume_checkpoint_filepath)):
        resume_training = True
    else:
        resume_training = False

valid_loss, valid_acc, valid_f1 = train_model(model=model,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              modalities=modalities,
                                              train_dataloader=train_dataloader,
                                              valid_dataloader=valid_dataloader,
                                              device=device,
                                              epochs=epochs_in_each_val_cycle,
                                              model_save_base_dir=model_save_base_dir,
                                              model_checkpoint_filename=f'{model_checkpoint_filename}',
                                              resume_checkpoint_filename=f'{resume_checkpoint_filename}',
                                              checkpoint_attribs=checkpoint_attribs,
                                              show_checkpoint_info=False,
                                              resume_training=resume_training,
                                              log_filename=log_filename,
                                              log_base_dir=log_base_dir,
                                              tensorboard_writer=tb_writer,
                                              strict_load=False,
                                              early_stop_patience=early_stop_patience)

result = f'{valid_acc}, {valid_f1}, final,' \
         f'{cnn_out_channel}, {kernel_size}, {feature_embed_size}, {lstm_hidden_size}, {lstm_encoder_num_layers},' \
         f'{lower_layer_dropout}, {upper_layer_dropout}, {module_embedding_nhead}, {multi_modal_nhead},' \
         f'{mm_embedding_attn_merge_type}' \
         f'{batch_size}, {lr}, {epochs_in_each_val_cycle}, {seq_max_len}, {window_size}, {window_stride}, ' \
         f'{data_dir_base_path}, {model_checkpoint_filename}, {log_base_dir}, {log_filename}\n'

log_execution(log_base_dir, final_log_filename, result, print_console=False)

log_execution(log_base_dir, log_filename,
              f'\n\n Final average test accuracy:{valid_acc}, avg f1_score {valid_f1} \n\n')
