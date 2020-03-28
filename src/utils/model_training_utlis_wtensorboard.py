import statistics

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .log import *
from . import config
from torch.utils.tensorboard import SummaryWriter

def load_model(model=None,
               optimizer=None,
               model_save_base_dir=None,
               model_checkpoint_filename=None,
               checkpoint_attribs=None,
               show_checkpoint_info=True,
               strict_load=True):
    print('strict_load',strict_load)
    if model == None or model_save_base_dir == None or model_checkpoint_filename == None or checkpoint_attribs == None:
        print('Missing load model parameters')
        return None, None

    model_checkpoint = torch.load(model_save_base_dir + '/' + model_checkpoint_filename)
    model.load_state_dict(model_checkpoint['state_dict'], strict=strict_load)
    if (optimizer != None):
        optimizer.load_state_dict(model_checkpoint['optimizer_state'])

    attrib_dict = {}
    for attrib in checkpoint_attribs:
        attrib_dict[attrib] = model_checkpoint[attrib]

    if (show_checkpoint_info):
        print('======Saved Model Info======')
        for attrib in checkpoint_attribs:
            print(attrib, model_checkpoint[attrib])
        print('=======Model======')
        print(model)
        print('#######Model#######')
        if (optimizer != None):
            print('=======Model======')
            print(optimizer)
            print('#######Model#######')

    print(f'loaded the model and optimizer successfully from {model_checkpoint_filename}')

    return model, optimizer, attrib_dict

def test_model_mu(model, optimizer, valid_dataloader,
                     loss_function, device,
                     modalities,
                     model_save_base_dir,
                     model_checkpoint_filename,
                     checkpoint_attribs, validation_iteration,
                     log_base_dir,
                     log_filename,
                     show_checkpoint_info=False,
                     is_ntu = False):
    valid_loss = 0.0
    valid_acc = 0.0
    valid_corrects = 0.0
    f1_scores = []
    preds_all = np.zeros(0)
    targets_all = np.zeros(0)
    attn_weights = {}
    seq_len = {}

    model, optimizer, attrib_dict = load_model(model=model, optimizer=optimizer,
                                  model_save_base_dir=model_save_base_dir,
                                  model_checkpoint_filename=model_checkpoint_filename,
                                  checkpoint_attribs=checkpoint_attribs, show_checkpoint_info=show_checkpoint_info)
    model.to(device)
    model.eval()
    with torch.no_grad():
        print('$$$$$ Start Testing Model $$$$$\n')
        for batch_idx, batch in enumerate(valid_dataloader):

            mask_graph = dict()
            tm_len = 0
            for modality in modalities:
                batch[modality] = batch[modality].to(device)
                batch[modality + '_mask'] = batch[modality + '_mask'].to(device)
                mask_graph[modality] = batch['modality_mask'].to(device)
                tm_len = max(tm_len, batch[modality].size(1))

            if(is_ntu):
                    batch['indi_sk_mask'] = batch['indi_sk_mask'].to(device)

            batch['modality_mask'] = batch['modality_mask'].to(device)
            batch['modality_mask_graph'] = mask_graph
            batch['label'] = batch['label'].to(device)
            labels = batch['label']
            
            outputs, module_attn_weights, mm_attn_weight  = model(batch)
            _, preds = torch.max(outputs, 1)

            true_label_index = labels.cpu().data.numpy()[0]
                
            if(true_label_index not in attn_weights.keys()):
                attn_weights[true_label_index] = {'mm_attn_weight': mm_attn_weight,
                                                  'module_attn_weights': module_attn_weights}
                seq_len[true_label_index] = tm_len
            elif(seq_len[true_label_index]>tm_len):
                attn_weights[true_label_index] = {'mm_attn_weight': mm_attn_weight,
                                                  'module_attn_weights': module_attn_weights}
                seq_len[true_label_index] = tm_len
            
            valid_corrects += torch.sum(preds == labels.data)
            f1_scores.append(f1_score(preds.cpu().data.numpy(), labels.cpu().data.numpy(), average='micro'))
            preds_all = np.append(preds_all, preds.cpu().data.numpy())
            targets_all = np.append(targets_all, labels.cpu().data.numpy())

            loss = loss_function(outputs, labels)
            valid_loss += loss.item()
            
            del batch
            torch.cuda.empty_cache()

    valid_loss = valid_loss / len(valid_dataloader.dataset)
    valid_acc = valid_corrects / len(valid_dataloader.dataset)
    print('Valid it[{}] Avg loss: {:.5f}, Acc:{:.5f}, F1:{:.5f}'.format(validation_iteration, valid_loss, valid_acc,
                                                                        statistics.mean(f1_scores)))
    log_execution(log_base_dir, log_filename,
                  '\n\n#####> Valid Avg loss: {:.5f}, Acc:{:.5f}, F1: {:.5f}\n\n'.format(valid_loss, valid_acc,
                                                                                         statistics.mean(f1_scores)))

    return valid_acc, statistics.mean(f1_scores), preds_all, targets_all, attn_weights

def test_model(model, optimizer, valid_dataloader,
                     loss_function, device,
                     modalities,
                     model_save_base_dir,
                     model_checkpoint_filename,
                     checkpoint_attribs, validation_iteration,
                     log_base_dir,
                     log_filename,
                     show_checkpoint_info=False,
                     is_ntu = False):
    valid_loss = 0.0
    valid_acc = 0.0
    valid_corrects = 0.0
    f1_scores = []
    preds_all = np.zeros(0)
    targets_all = np.zeros(0)
    attn_weights = {}
    seq_len = {}

    model, optimizer, attrib_dict = load_model(model=model, optimizer=optimizer,
                                  model_save_base_dir=model_save_base_dir,
                                  model_checkpoint_filename=model_checkpoint_filename,
                                  checkpoint_attribs=checkpoint_attribs, show_checkpoint_info=show_checkpoint_info)
    model.to(device)
    model.eval()
    with torch.no_grad():
        print('$$$$$ Start Testing Model $$$$$\n')
        for batch_idx, batch in enumerate(valid_dataloader):

            mask_graph = dict()
            tm_len = 0
            for modality in modalities:
                batch[modality] = batch[modality].to(device)
                batch[modality + '_mask'] = batch[modality + '_mask'].to(device)
                mask_graph[modality] = batch['modality_mask'].to(device)
                tm_len = max(tm_len, batch[modality].size(1))

            if(is_ntu):
                    batch['indi_sk_mask'] = batch['indi_sk_mask'].to(device)

            batch['modality_mask'] = batch['modality_mask'].to(device)
            batch['modality_mask_graph'] = mask_graph
            batch['label'] = batch['label'].to(device)
            labels = batch['label']

            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            
            true_label_index = labels.cpu().data.numpy()[0]
                
            if(true_label_index not in attn_weights.keys()):
                attn_weights[true_label_index] = {'mm_attn_weight': model.mm_attn_weight,
                                                  'module_attn_weights': model.module_attn_weights}
                seq_len[true_label_index] = tm_len
            elif(seq_len[true_label_index]>tm_len):
                attn_weights[true_label_index] = {'mm_attn_weight': model.mm_attn_weight,
                                                  'module_attn_weights': model.module_attn_weights}
                seq_len[true_label_index] = tm_len

            valid_corrects += torch.sum(preds == labels.data)
            f1_scores.append(f1_score(preds.cpu().data.numpy(), labels.cpu().data.numpy(), average='micro'))
            preds_all = np.append(preds_all, preds.cpu().data.numpy())
            targets_all = np.append(targets_all, labels.cpu().data.numpy())

            loss = loss_function(outputs, labels)
            valid_loss += loss.item()
            
            del batch
            torch.cuda.empty_cache()

    valid_loss = valid_loss / len(valid_dataloader.dataset)
    valid_acc = valid_corrects / len(valid_dataloader.dataset)
    print('Valid it[{}] Avg loss: {:.5f}, Acc:{:.5f}, F1:{:.5f}'.format(validation_iteration, valid_loss, valid_acc,
                                                                        statistics.mean(f1_scores)))
    log_execution(log_base_dir, log_filename,
                  '\n\n#####> Valid Avg loss: {:.5f}, Acc:{:.5f}, F1: {:.5f}\n\n'.format(valid_loss, valid_acc,
                                                                                         statistics.mean(f1_scores)))

    return valid_acc, statistics.mean(f1_scores), preds_all, targets_all, attn_weights


def model_validation(model, optimizer, valid_dataloader,
                     loss_function, device,
                     modalities,
                     model_save_base_dir,
                     model_checkpoint_filename,
                     checkpoint_attribs,
                     log_base_dir,
                     log_filename,
                     show_checkpoint_info=False,
                     is_load=False,
                     strict_load=True):
    valid_loss = 0.0
    valid_corrects = 0.0
    f1_scores = []

    if(is_load):
        model, optimizer, attrib_dict = load_model(model=model, optimizer=optimizer,
                                          model_save_base_dir=model_save_base_dir,
                                          model_checkpoint_filename=model_checkpoint_filename,
                                          checkpoint_attribs=checkpoint_attribs,
                                          show_checkpoint_info=show_checkpoint_info,
                                          strict_load=strict_load)

    model.eval()
    for batch_idx, batch in enumerate(valid_dataloader):
        
        mask_graph = dict()
        for modality in modalities:
            batch[modality] = batch[modality].to(device)
            batch[modality + config.modality_seq_len_tag] = batch[modality + config.modality_seq_len_tag].to(device)
            batch[modality + config.modality_mask_suffix_tag] = batch[modality + config.modality_mask_suffix_tag].to(device)
            mask_graph[modality] = batch['modality_mask'].to(device)

        batch['modality_mask'] = batch['modality_mask'].to(device)
        batch['modality_mask_graph'] = mask_graph
        batch['label'] = batch['label'].to(device)
        labels = batch['label']

        outputs = model(batch)
        _, preds = torch.max(outputs, 1)

        valid_corrects += torch.sum(preds == labels.data)
        f1_scores.append(f1_score(preds.cpu().data.numpy(), labels.cpu().data.numpy(), average='micro'))

        loss = loss_function(outputs, labels)
        valid_loss += loss.item()
        
        del batch
        torch.cuda.empty_cache()

    valid_loss = valid_loss / len(valid_dataloader.dataset)
    valid_acc = valid_corrects / len(valid_dataloader.dataset)
    log_execution(log_base_dir, log_filename,
                  '#####> Valid Avg loss: {:.5f}, Acc:{:.5f}, F1: {:.5f}\n'.format(valid_loss, valid_acc,
                                                                                 statistics.mean(f1_scores)))
    del model
    del optimizer
    torch.cuda.empty_cache()
    
    return valid_loss, valid_acc, statistics.mean(f1_scores)


def train_model(model, optimizer, scheduler,
                modalities,
                train_dataloader,
                valid_dataloader,
                device,
                model_save_base_dir,
                model_checkpoint_filename,
                resume_checkpoint_filename,
                checkpoint_attribs,
                log_base_dir,
                log_filename,
                epochs=100,
                resume_training=True,
                show_checkpoint_info=False,
                improvement_val_it=100,
                strict_load=True,
                tensorboard_writer=None,
                early_stop_patience=50):

    model.to(device)

    train_loss_min = np.Inf
    train_acc_max = 0.0
    valid_loss_min = np.Inf
    valid_acc_max = 0.0
    f1_prev = 0.0
    valid_f1_max = 0.0
    train_loss_th = 1e-5
    train_acc_th = 1.0
    start_epoch = 1
    early_stop_counter = 0

    if (resume_training):
        model, optimizer, attrib_dict = load_model(model=model, optimizer=optimizer,
                                      model_save_base_dir=model_save_base_dir,
                                      model_checkpoint_filename=resume_checkpoint_filename,
                                      checkpoint_attribs=checkpoint_attribs,
                                      show_checkpoint_info=show_checkpoint_info,
                                      strict_load=strict_load)
        valid_loss_min = float(attrib_dict['valid_loss'])
        train_loss_min = float(attrib_dict['train_loss'])
        train_acc_max = float(attrib_dict['train_acc'])
        valid_acc_max = float(attrib_dict['valid_acc'])
        start_epoch = max(1, int(attrib_dict['epoch'] + 1))

        log_execution(log_base_dir, log_filename,
                      f'resume training from {start_epoch} and previous valid_loss_min {valid_loss_min}, train_loss_min {train_loss_min}\n')
        log_execution(log_base_dir, log_filename,
                      f'previous valid_loss_min {valid_loss_min}, train_loss_min {train_loss_min}\n')

        if (valid_loss_min == 0):
            valid_loss_min = np.Inf
            log_execution(log_base_dir, log_filename, f'valid loss set to {valid_loss_min}\n')
        if (train_loss_min == 0):
            train_loss_min = np.Inf
            log_execution(log_base_dir, log_filename, f'train loss set to {train_loss_min}\n')

        log_execution(log_base_dir, log_filename,
                      f'Resume training successfully from resume chekpoint filename: {resume_checkpoint_filename}\n model_checkpoint_filename {model_checkpoint_filename}\n')

    loss_function = nn.CrossEntropyLoss()
    improvement_it = 0
    train_dataloader_len = len(train_dataloader)

    tm_valid_acc_max = valid_acc_max
    tm_valid_loss_min = valid_loss_min
    tm_valid_f1_max = valid_f1_max
    valid_log_it = 0

    batch_train_acc_max = 0
    batch_train_loss_min = np.Inf

    batch_valid_acc_max = 0
    batch_valid_loss_min = np.Inf
    batch_valid_f1_max = 0

    for epoch in tqdm(range(start_epoch, epochs + 1)):

        train_loss = 0.0
        train_corrects = 0.0
        f1_scores = []

        model.train()

        batch_improvement_it = 0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            mask_graph = dict()
            for modality in modalities:
                batch[modality] = batch[modality].to(device)
                batch[modality + config.modality_seq_len_tag] = batch[modality + config.modality_seq_len_tag].to(device)
                batch[modality + config.modality_mask_suffix_tag] = batch[modality + config.modality_mask_suffix_tag].to(device)
                mask_graph[modality] = batch['modality_mask'].to(device)

            batch['modality_mask'] = batch['modality_mask'].to(device)
            batch['modality_mask_graph'] = mask_graph
            batch['label'] = batch['label'].to(device)
            labels = batch['label']
            batch_size = batch['label'].size(0)

            outputs = model(batch)
            _, preds = torch.max(outputs, 1)

            batch_corrects = torch.sum(preds == labels.data)
            batch_train_acc = batch_corrects / batch_size
            train_corrects += batch_corrects
            f1_scores.append(f1_score(preds.cpu().data.numpy(), labels.cpu().data.numpy(), average='micro'))

            loss = loss_function(outputs, labels)
            train_loss += loss.item()
            batch_loss = loss.item()/len(batch)

            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / train_dataloader_len)
            
            del batch
            torch.cuda.empty_cache() 

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\tAcc: {:.4f}'.format(
                    epoch, (batch_idx+1) * batch_size, len(train_dataloader),
                           100.0 * float(batch_idx) / float(len(train_dataloader)),
                    batch_loss, batch_train_acc))

            if(batch_train_acc > batch_train_acc_max):
                batch_improvement_it += 1
                batch_train_acc_max = batch_train_acc
                if(batch_improvement_it>improvement_val_it):
                    batch_improvement_it=0
                    batch_valid_loss, batch_valid_acc, batch_valid_f1 = model_validation(model=model, optimizer=optimizer,
                                                                       valid_dataloader=valid_dataloader,
                                                                       loss_function=loss_function,
                                                                       device=device,
                                                                       modalities=modalities,
                                                                       model_save_base_dir=model_save_base_dir,
                                                                       model_checkpoint_filename=model_checkpoint_filename,
                                                                       checkpoint_attribs=checkpoint_attribs,
                                                                       show_checkpoint_info=show_checkpoint_info,
                                                                       log_base_dir=log_base_dir,
                                                                       log_filename=log_filename,
                                                                       is_load=False)

                    if (tensorboard_writer):
                        tensorboard_writer.add_scalar(config.tbw_valid_loss, {'batch:':valid_loss}, valid_log_it)
                        tensorboard_writer.add_scalar(config.tbw_valid_acc, {'batch:':valid_acc}, valid_log_it)
                        tensorboard_writer.add_scalar(config.tbw_valid_f1, {'batch:':valid_f1}, valid_log_it)
                        valid_log_it += 1

                    if(batch_valid_acc_max > batch_valid_acc):
                        checkpoint = {'epoch': epoch,
                                      'state_dict': model.state_dict(),
                                      'optimizer_state': optimizer.state_dict(),
                                      'train_loss': train_loss,
                                      'valid_loss': valid_loss,
                                      'train_acc': train_acc,
                                      'valid_acc': valid_acc}

                        torch.save(checkpoint, f'{model_save_base_dir}/best_batch_valid_acc_{model_checkpoint_filename}')
                        log_execution(log_base_dir, log_filename,
                                      '\n####> Epoch: {}, batch{}: batch validation acc increased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                                          epoch, batch_idx, batch_valid_loss_min, batch_valid_loss, batch_valid_acc_max, batch_valid_acc, batch_valid_f1_max,
                                          batch_valid_f1))
                        log_execution(log_base_dir, log_filename,
                                      f'Best batch valid model save to best_{model_checkpoint_filename}\n')

                        batch_valid_f1_max = batch_valid_f1
                        batch_valid_acc_max = batch_valid_acc

                    if (batch_valid_loss_min > batch_valid_loss):
                        checkpoint = {'epoch': epoch,
                                      'state_dict': model.state_dict(),
                                      'optimizer_state': optimizer.state_dict(),
                                      'train_loss': train_loss,
                                      'valid_loss': valid_loss,
                                      'train_acc': train_acc,
                                      'valid_acc': valid_acc}

                        torch.save(checkpoint, f'{model_save_base_dir}/best_batch_valid_loss_{model_checkpoint_filename}')
                        log_execution(log_base_dir, log_filename,
                                      '\n####> Epoch: {}, batch{}: batch validation loss decreased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                                          epoch, batch_idx, batch_valid_loss_min, batch_valid_loss, batch_valid_acc_max,
                                          batch_valid_acc, batch_valid_f1_max,
                                          batch_valid_f1))
                        log_execution(log_base_dir, log_filename,
                                      f'Best batch valid model save to best_{model_checkpoint_filename}\n')

                        batch_valid_f1_max = batch_valid_f1
                        batch_valid_loss_min = batch_valid_loss
               

        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_corrects / len(train_dataloader.dataset)
        train_f1 = statistics.mean(f1_scores)
        log_execution(log_base_dir, log_filename,'====> Epoch: {} Train Avg loss: {:.5f}, Acc: {:.5f}, F1: {:.5f}'.format(epoch, train_loss, train_acc,
                                                                                       train_f1))

        valid_loss, valid_acc, valid_f1 = model_validation(model=model, optimizer=optimizer,
                                                           valid_dataloader=valid_dataloader,
                                                           loss_function=loss_function,
                                                           device=device,
                                                           modalities=modalities,
                                                           model_save_base_dir=model_save_base_dir,
                                                           model_checkpoint_filename=model_checkpoint_filename,
                                                           checkpoint_attribs=checkpoint_attribs,
                                                           show_checkpoint_info=show_checkpoint_info,
                                                           log_base_dir=log_base_dir,
                                                           log_filename=log_filename,
                                                           is_load=False)
        if (tensorboard_writer):
            tensorboard_writer.add_scalar(config.tbw_train_loss, train_loss, epoch)
            tensorboard_writer.add_scalar(config.tbw_train_acc, train_acc, epoch)
            tensorboard_writer.add_scalar(config.tbw_train_f1, train_f1, epoch)
            print('tensor board log', epoch)

        if train_loss <= train_loss_min:
            log_execution(log_base_dir, log_filename,
                          '===> Epoch: {}: Training loss decreased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                              epoch, train_loss_min, train_loss, train_acc_max, train_acc, f1_prev,train_f1))

            train_loss_min = train_loss
            train_acc_max = train_acc
            f1_prev = statistics.mean(f1_scores)

            checkpoint = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'train_loss': train_loss,
                          'valid_loss': valid_loss,
                          'train_acc': train_acc,
                          'valid_acc': valid_acc}

            torch.save(checkpoint, f'{model_save_base_dir}/{model_checkpoint_filename}')
            log_execution(log_base_dir, log_filename, f'model saved to {model_checkpoint_filename}\n')
            improvement_it = improvement_it + 1

            if (valid_loss < valid_loss_min):
                early_stop_counter = 0
            else:
                early_stop_counter +=1
        
        if (tensorboard_writer):
            tensorboard_writer.add_scalar(config.tbw_valid_loss, valid_loss, epoch)
            tensorboard_writer.add_scalar(config.tbw_valid_acc, valid_acc, epoch)
            tensorboard_writer.add_scalar(config.tbw_valid_f1, valid_f1, epoch)

        if (valid_loss < valid_loss_min):
            checkpoint = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'train_loss': train_loss,
                          'valid_loss': valid_loss,
                          'train_acc': train_acc,
                          'valid_acc': valid_acc}

            torch.save(checkpoint, f'{model_save_base_dir}/best_valid_loss_{model_checkpoint_filename}')
            log_execution(log_base_dir, log_filename,
                          '\n####> Epoch: {}: validation loss decreased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                              epoch, valid_loss_min, valid_loss, valid_acc_max, valid_acc, valid_f1_max, valid_f1))
            log_execution(log_base_dir, log_filename, f'Best valid model save to best_{model_checkpoint_filename}\n')

            valid_f1_max = valid_f1
            valid_loss_min = valid_loss

            early_stop_counter = 0

        if (valid_acc > tm_valid_acc_max):
            checkpoint = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'train_loss': train_loss,
                          'valid_loss': valid_loss,
                          'train_acc': train_acc,
                          'valid_acc': valid_acc}

            torch.save(checkpoint, f'{model_save_base_dir}/best_acc_{model_checkpoint_filename}')
            log_execution(log_base_dir, log_filename,
                          '\n####> Epoch: {}: validation acc increase ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}), F1: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                              epoch, tm_valid_loss_min, valid_loss, tm_valid_acc_max, valid_acc, tm_valid_f1_max, valid_f1))
            log_execution(log_base_dir, log_filename, f'Best valid model (acc) save to best_acc_{model_checkpoint_filename}\n')

            tm_valid_f1_max = valid_f1
            tm_valid_acc_max = valid_acc

        if(early_stop_counter>early_stop_patience):
            log_execution(log_base_dir, log_filename, '\n##### Epoch: {}: Training cycle break due to early stop, patience{}\n'.format(epoch, early_stop_counter))
            break

        if (valid_loss < train_loss_th and valid_acc >= train_acc_th):
            log_execution(log_base_dir, log_filename,
                          '\n\n##### Epoch: {}: Training cycle break due to low loss{:.4f} and 1.0 accuracy\n\n'.format(
                              epoch, train_loss))
            break

    return valid_loss_min, valid_acc_max, valid_f1_max


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          rotation=45,
                         save_fig_dir='figures',
                         save_fig_name=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=rotation)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if(save_fig_name is not None):
        plt.savefig(f'{save_fig_dir}/{save_fig_name}', bbox_inches='tight', dpi=600)
        print(f'figure save to {save_fig_dir}/{save_fig_name}')
    plt.show()
