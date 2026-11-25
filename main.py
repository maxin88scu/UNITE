import argparse
from model import *
from utils import *
from sklearn.metrics import f1_score
import itertools
import random

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--nhid', type=int, default=512, help='hidden size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--known_class', type=int, default=3, help='number of known classes')
parser.add_argument('--device', type=str, default='cuda', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--source', type=str, default='ACMv9', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='Citationv1', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument("--lblToRemove1", type=int, default=0, help='first label to remove for source domain data')
parser.add_argument("--lblToRemove2", type=int, default=1, help='second label to remove for source domain data')
parser.add_argument("--topk", type=int, default=2, help="the number of topk elements")
parser.add_argument("--beta", type=float, default=0.99, help="none")
parser.add_argument("--gamma", type=float, default=0.02, help="none")
parser.add_argument("--temperature", type=float, default=0.5, help="temperature for contrastive learning")
parser.add_argument("--times", type=int, default=10, help="run times")
parser.add_argument("--weight", type=float, default=0.2, help="weight")
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')

args = parser.parse_args()

criterion = nn.CrossEntropyLoss().to(args.device)

def train(epoch,models,optimizer,source_data,target_data,unknown_detector):
    for model in models:
        model.train()
    model1 = models[0]
    model2 = models[1]

    encoded_source1 = model1.feat_bottleneck(source_data.x, source_data.edge_index)
    encoded_target1 = model1.feat_bottleneck(target_data.x, target_data.edge_index)
    logits_source1 = model1.feat_classifier(encoded_source1, source_data.edge_index)
    logits_target1 = model1.feat_classifier(encoded_target1, target_data.edge_index)

    encoded_source2 = model2.feat_bottleneck(source_data.x, source_data.edge_index)
    encoded_target2 = model2.feat_bottleneck(target_data.x, target_data.edge_index)

    logits_source2 = model2.feat_classifier(encoded_source2, source_data.edge_index)[:, :args.known_class]
    logits_target2 = model2.feat_classifier(encoded_target2, target_data.edge_index)[:, :args.known_class]

    warm_up = 20
    labels = source_data.y

    if epoch <= warm_up:
        loss_s_cls1 = criterion(logits_source1, labels)
        loss_s_cls2 = criterion(logits_source2, labels)
        loss = loss_s_cls1 + loss_s_cls2
    else:
        known_index, unknown_index = unknown_detector(logits_source2, logits_target2, target_data.edge_index)

        neighbors = get_neighbors(logits_source1, logits_target1, args.topk, args.device)
        loss_ncl = neighbor_contrastive_loss(encoded_source1, encoded_target1, neighbors, args.temperature)


        loss_s_cls1 = criterion(logits_source1, labels)
        loss_s_cls2 = criterion(logits_source2, labels)

        if known_index[0].size(0) == 0:
            loss_t_cls = 0
        else:
            target_pseudo_label = logits_target1.argmax(dim=1)
            target_pseudo_label[unknown_index] = 3
            loss_t_cls = criterion(logits_target1, target_pseudo_label)
        loss_cls = loss_s_cls1 + loss_s_cls2 + loss_t_cls * float(args.weight)

        loss_nc = 0.0
        if unknown_index[0].numel() > 0:
            loss_nc = neighbor_center_contrastive_loss(encoded_source1, encoded_target1, unknown_index, labels, args.device, args.temperature)

        loss = loss_ncl + loss_cls + loss_nc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def target_eval(model,target_data):
    model.eval()

    with torch.no_grad():
        encoded_target = model.feat_bottleneck(target_data.x, target_data.edge_index)
        target_logits = model.feat_classifier(encoded_target, target_data.edge_index)

        target_predictions = target_logits.argmax(dim=1)

        known_index = (target_data.y < 3).nonzero().squeeze()
        unknown_index = (target_data.y == 3).nonzero().squeeze()

        all_average_acc1 = f1_score(target_data.y.cpu().detach(), target_predictions.cpu().detach(), average='micro')

        known_average_acc1 = f1_score(target_data.y[known_index].cpu().detach(),
                                      target_predictions[known_index].cpu().detach(), average='micro')
        unknown_average_acc1 = target_predictions[unknown_index].eq(target_data.y[unknown_index]).float().mean()
        H_score1 = 2 * known_average_acc1 * unknown_average_acc1 / (known_average_acc1 + unknown_average_acc1)

    return all_average_acc1, known_average_acc1, unknown_average_acc1, H_score1


def main():

    label_list = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

    every_best_acc=[]
    every_best_known_acc = []
    every_best_unknown_acc = []
    every_best_H_score = []

    for i in range(args.times):

        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))


        args.lblToRemove1 = int(label_list[i][0])
        args.lblToRemove2 = int(label_list[i][1])

        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.source)
        source_dataset = CitationDataset(path, args.source, domain='source', lblToRemove1=args.lblToRemove1,
                                         lblToRemove2=args.lblToRemove2)

        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.target)
        target_dataset = CitationDataset(path, args.target, domain='target', lblToRemove1=args.lblToRemove1,
                                         lblToRemove2=args.lblToRemove2)

        target_data = target_dataset[0].to(args.device)
        source_data = source_dataset[0].to(args.device)

        print("load data down!")

        args.num_features = source_data.x.size(1)

        print("the {} experiment".format(i))
        print("args:",args)

        model1 = Model(args).to(args.device)
        model2 = Model(args).to(args.device)
        models = [model1, model2]
        params = itertools.chain(*[model.parameters() for model in models])
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        unknown_detector = UnknownDetector(source_data.y, args.known_class, args.beta, args.gamma)


        best_acc = 0.0
        best_known_acc = 0.0
        best_unknown_acc = 0.0
        best_H_score = 0.0


        for epoch in range(1, args.epochs + 1):
            train(epoch,models,optimizer,source_data,target_data,unknown_detector)
            acc, known_acc, unknown_acc, H_score = target_eval(model1,target_data)

            if epoch % 10 == 0:
                print("epoch:{}, acc:{:.4f}, known_acc:{:.4f}, unknown_acc:{:.4f}, H_score:{:.4f}".format(epoch, acc, known_acc,unknown_acc,H_score))

            if acc > best_acc:
                best_H_score = H_score
                best_known_acc = known_acc
                best_unknown_acc = unknown_acc
                best_acc = acc
        print("-------------------------------------------------------")
        print("source:{},target:{},remove_label:{}/{}, best acc:{:.4f}, best known_acc:{:.4f}, best unknown_acc:{:.4f}, best H_score:{:.4f}".format(args.source,args.target,args.lblToRemove1,args.lblToRemove2, best_acc, best_known_acc, best_unknown_acc, best_H_score))

        every_best_acc.append(best_acc)
        every_best_known_acc.append(best_known_acc)
        every_best_unknown_acc.append(best_unknown_acc)
        every_best_H_score.append(best_H_score)

    every_best_acc_np = np.array(every_best_acc)
    every_best_H_score_np = np.array([tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor for tensor in every_best_H_score])

    mean_acc = np.mean(every_best_acc_np)
    std_acc = np.std(every_best_acc_np)
    mean_h_score = np.mean(every_best_H_score_np)
    std_h_score = np.std(every_best_H_score_np)

    print("===============================================================================================\n")
    print("source:{},target:{},task done!".format(args.source,args.target))
    print("best_acc_every_time:")
    for acc in every_best_acc:
        print("acc:{:.4f}".format(acc))
    print("best_h_score_every_time:")
    for h_score in every_best_H_score:
        print("h_score:{:.4f}".format(h_score))

    print("best_known_acc_every_time:")
    for known_acc in every_best_known_acc:
        print("acc:{:.4f}".format(known_acc))

    print("best_unknown_acc_every_time:")
    for unknown_acc in every_best_unknown_acc:
        print("acc:{:.4f}".format(unknown_acc))

    print("mean acc:",mean_acc)
    print("std acc:",std_acc)
    print("mean h_score:",mean_h_score)
    print("std h_score:",std_h_score)

if __name__ == '__main__':
    main()
