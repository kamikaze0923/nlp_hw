from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
import argparse
import os, shutil
from udpos.model import POS_from_WordSeq
from udpos.dataset import create_torch_UDPOS_dataset_and_embedding_layer
from udpos.rountine import routine, pad_collate
from udpos.visualize import plot_loss, plot_confusion_matrix, to_gif, Training_Info_Buffer


def main(args):
    datasets, word_embedding_layer, tag_embedding_layer = create_torch_UDPOS_dataset_and_embedding_layer(args)
    print("Length of datasets train {}, valid {}, test {}".format(len(datasets[0]), len(datasets[1]), len(datasets[2])))
    collate_with_args = lambda b: pad_collate(b, args)
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_args)
    valid_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_args)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_args)

    device = 'cuda:0' if args.cuda else 'cpu'
    pos_model = POS_from_WordSeq(args, word_embedding_layer, tag_embedding_layer).to(device)
    adam_opt = Adam(params=pos_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

    for name, param in pos_model.named_parameters(): # move everything to GPU here
        if param.requires_grad:
            print(f"need gradient on {param.device} {name}")
        else:
            print(f"does not need gradient {param.device} {name}")

    training_info = f"{args.seed}-{args.batch_size}-{args.epochs}-{args.learning_rate}"
    model_info = f"{args.hidden_dim}-{args.lstm_layers}-{args.embedding_dim}-{args.use_encoder}"
    hyper_parameters = f"bi_lstm-" + training_info + "-" + model_info

    try:
        shutil.rmtree(hyper_parameters)
    except:
        pass

    ckpt_dir = os.path.join(hyper_parameters, args.save_folder)
    os.makedirs(ckpt_dir, exist_ok=True)
    animate_dir = "animation"
    train_image_dir = os.path.join(hyper_parameters, animate_dir, "train")
    valid_image_dir = os.path.join(hyper_parameters, animate_dir, "valid")
    for dir in [train_image_dir, valid_image_dir]:
        os.makedirs(dir, exist_ok=True)

    info_buffer = Training_Info_Buffer()
    best_acc = 0
    best_epoch = 0
    for e in range(args.epochs):
        print("Epoch: {}".format(e))
        train_loss, train_acc, train_word_stats, cfs_MT = routine(train_loader, pos_model, optimizer=adam_opt)
        plot_confusion_matrix(cfs_MT, label_dict=datasets[0].tag_label, save_dir=train_image_dir, epoch=e)
        valid_loss, valid_acc, valid_word_stats, cfs_MV = routine(valid_loader, pos_model, optimizer=None)
        plot_confusion_matrix(cfs_MV, label_dict=datasets[1].tag_label, save_dir=valid_image_dir, epoch=e)

        print("Epoch train loss: {:.5f}, train acc: {:.3f}({}/{}), valid loss: {:.5f}, valid acc: {:.3f}({}/{})".format(
            train_loss, train_acc, train_word_stats[0], train_word_stats[1],
            valid_loss, valid_acc, valid_word_stats[0], valid_word_stats[1])
        )
        info_buffer.train_loss_buffer.append(train_loss)
        info_buffer.validate_loss_buffer.append(valid_loss)
        info_buffer.train_acc_buffer.append(train_acc)
        info_buffer.validate_acc_buffer.append(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = e

        if not args.debug and False:
            torch.save(pos_model.state_dict(), os.path.join(ckpt_dir, "ckpt_{}.pt".format(e)))
        # print("Unk embedding {}".format(pos_model.encoder.word_embedding_layer.unk_parameter))

    test_loss, test_acc, _, _ = routine(test_loader, pos_model, optimizer=None)
    plot_loss(hyper_parameters, info_buffer, (test_loss, test_acc), (best_acc, best_epoch), args)
    to_gif(hyper_parameters, train_image_dir, valid_image_dir, name="cfm")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Number of hidden units in LSTM.')
    parser.add_argument('--lstm-layers', type=int, default=1,
                        help='Number of hidden units in LSTM.')
    parser.add_argument('--embedding-dim', type=int, default=300,
                        help='Dimensionality of embedding.')
    parser.add_argument('--use-encoder', action='store_true', default=False,
                        help='Using a biLSTM encoder')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Reduce the dataset for faster local debugging')
    parser.add_argument('--debug-dataset-size', type=int, default=20,
                        help='Use a tiny dataset to debug the program first')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.debug:
        args.batch_size = 100
        args.epochs = 100
        args.hidden_dim = 64
        args.embedding_dim = 300
        args.debug_dataset_size = 2000

    if args.cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    for k,v in vars(args).items():
        print(f"{k}: {v}")
    torch.manual_seed(args.seed)
    main(args)

