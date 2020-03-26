import statistics

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .log import *
from . import config
from torch.utils.tensorboard import SummaryWriter

from .losses import SimLoss


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

def model_validation(model, optimizer, valid_dataloader,
                     classification_loss,
                     simLoss,
                     device,
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
    total_samples = 0.0
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
        batch_size = batch['label'].size(0)
        total_samples += batch_size

        outputs,embeds = model(batch)
        _, real_preds = torch.max(outputs[config.real_modality_tag], 1)
        _, fake_preds = torch.max(outputs[config.fake_modality_tag], 1)

        real_labels = torch.ones_like(labels).data
        fake_labels = torch.zeros_like(labels).data
        batch_corrects = torch.sum(real_preds == real_labels)
        batch_corrects += torch.sum(fake_preds == fake_labels)

        batch_valid_acc = batch_corrects / (2.0*batch_size)
        valid_corrects += batch_corrects
        f1_scores.append(f1_score(real_preds.cpu().data.numpy(), real_labels.cpu().numpy(), average='micro'))
        f1_scores.append(f1_score(fake_preds.cpu().data.numpy(), fake_labels.cpu().numpy(), average='micro'))

        real_loss = classification_loss(outputs[config.real_modality_tag], labels)
        fake_loss = classification_loss(outputs[config.fake_modality_tag], labels)
        sim_loss = simLoss(embeds[config.real_modality_tag],embeds[config.fake_modality_tag])
        loss =sim_loss+real_loss+fake_loss

        valid_loss += loss.item()
        batch_loss = loss.item()/batch_size        
        
        del batch
        torch.cuda.empty_cache()

    valid_loss = valid_loss / len(valid_dataloader.dataset)
    valid_acc = valid_corrects / (2.0 *total_samples)
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

    classification_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.35,0.35],
                                                                   dtype=torch.float,
                                                                  device=device))
    simLoss = SimLoss(weight=0.3)
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
        batch_improvement_it = 0
        total_samples = 0.0
        
        model.train()
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
            total_samples += batch_size

            outputs,embeds = model(batch)
            _, real_preds = torch.max(outputs[config.real_modality_tag], 1)
            _, fake_preds = torch.max(outputs[config.fake_modality_tag], 1)

            real_labels = torch.ones_like(labels).data
            fake_labels = torch.zeros_like(labels).data
            batch_corrects = torch.sum(real_preds == real_labels)
            batch_corrects += torch.sum(fake_preds == fake_labels)

            batch_train_acc = batch_corrects / (2.0*batch_size)
            train_corrects += batch_corrects
            f1_scores.append(f1_score(real_preds.cpu().data.numpy(), real_labels.cpu().numpy(), average='micro'))
            f1_scores.append(f1_score(fake_preds.cpu().data.numpy(), fake_labels.cpu().numpy(), average='micro'))

            real_loss = classification_loss(outputs[config.real_modality_tag], labels)
            fake_loss = classification_loss(outputs[config.fake_modality_tag], labels)
            sim_loss = simLoss(embeds[config.real_modality_tag],embeds[config.fake_modality_tag])
            loss =sim_loss+real_loss+fake_loss

            train_loss += loss.item()
            batch_loss = loss.item()/batch_size

            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / train_dataloader_len)
            
            del batch
            torch.cuda.empty_cache() 

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\tAcc: {:.4f}'.format(
                    epoch, (batch_idx+1), len(train_dataloader),
                           100.0 * float(batch_idx) / float(len(train_dataloader)),
                    batch_loss, batch_train_acc))

            if(batch_train_acc > batch_train_acc_max):
                batch_improvement_it += 1
                batch_train_acc_max = batch_train_acc
                if(batch_improvement_it>improvement_val_it):
                    batch_improvement_it=0
                    batch_valid_loss, batch_valid_acc, batch_valid_f1 = model_validation(model=model, optimizer=optimizer,
                                                                       valid_dataloader=valid_dataloader,
                                                                       classification_loss=classification_loss,
                                                                       simLoss = simLoss,
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
                                      'train_loss': 0,
                                      'valid_loss': batch_valid_loss,
                                      'train_acc': 0,
                                      'valid_acc': batch_valid_acc}

                        torch.save(checkpoint, f'{model_save_base_dir}/best_batch_valid_acc_{model_checkpoint_filename}')
                        log_execution(log_base_dir, log_filename,
                                      '\n####> Epoch: {}, batch{}: batch validation acc increased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                                          epoch, batch_idx, batch_valid_loss_min, batch_valid_loss, batch_valid_acc_max, batch_valid_acc))
                        log_execution(log_base_dir, log_filename,
                                      f'Best batch valid model save to best_{model_checkpoint_filename}\n')

                        batch_valid_acc_max = batch_valid_acc

                    if (batch_valid_loss_min > batch_valid_loss):
                        checkpoint = {'epoch': epoch,
                                      'state_dict': model.state_dict(),
                                      'optimizer_state': optimizer.state_dict(),
                                      'train_loss': 0,
                                      'valid_loss': batch_valid_loss,
                                      'train_acc': 0,
                                      'valid_acc': batch_valid_acc}

                        torch.save(checkpoint, f'{model_save_base_dir}/best_batch_valid_loss_{model_checkpoint_filename}')
                        log_execution(log_base_dir, log_filename,
                                      '\n####> Epoch: {}, batch{}: batch validation loss decreased ({:.5f} --> {:.5f}), Acc: ({:.5f} --> {:.5f}).  Saving model ...\n'.format(
                                          epoch, batch_idx, batch_valid_loss_min, batch_valid_loss, batch_valid_acc_max,
                                          batch_valid_acc))
                        log_execution(log_base_dir, log_filename,
                                      f'Best batch valid model save to best_{model_checkpoint_filename}\n')

                        batch_valid_loss_min = batch_valid_loss
               

        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_corrects / (2. * total_samples)
        train_f1 = statistics.mean(f1_scores)
        log_execution(log_base_dir, log_filename,'====> Epoch: {} Train Avg loss: {:.5f}, Acc: {:.5f}, F1: {:.5f}'.format(epoch, train_loss, train_acc,
                                                                                       train_f1))
        
        valid_loss, valid_acc, valid_f1 = model_validation(model=model, optimizer=optimizer,
                                                           valid_dataloader=valid_dataloader,
                                                           classification_loss=classification_loss,
                                                           simLoss=simLoss,
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
            early_stop_counter = 0

        if(early_stop_counter>early_stop_patience):
            log_execution(log_base_dir, log_filename, '\n##### Epoch: {}: Training cycle break due to early stop, patience{}\n'.format(epoch, early_stop_counter))
            break

        if (valid_loss < train_loss_th and valid_acc >= train_acc_th):
            log_execution(log_base_dir, log_filename,
                          '\n\n##### Epoch: {}: Training cycle break due to low loss{:.4f} and 1.0 accuracy\n\n'.format(
                              epoch, train_loss))
            break

    return valid_loss_min, valid_acc_max, valid_f1_max