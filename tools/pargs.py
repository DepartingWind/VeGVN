import argparse

def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], help='Running mode: train or test')
    parser.add_argument('--dataset', type=str, default='WeiboCED')
    parser.add_argument('--dataset_tree', type=str, default='CED_tree_OCRd')
    parser.add_argument('--dataset_text', type=str, default='CED_tree_text_OCRd')
    parser.add_argument('--dataset_label', type=str, default='CED_label')
    parser.add_argument('--dataset_pic', type=str, default='CED_label')
    parser.add_argument('--modelname', type=str, default='VeGCN')
    parser.add_argument('--Gpath', type=str, default='graph')

    parser.add_argument('--fusion', type=str, default='CoSelf')
    parser.add_argument('--lambd', type=int, default=1, help='')

    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--seed', type=int, default=66)

    parser.add_argument('--is_vision_graph', type=str2bool, default=True)
    parser.add_argument('--base_config', type=str2bool, default=False)

    parser.add_argument('--vector_size', type=int, help='word embedding size', default=300)
    parser.add_argument('--image_size', type=int, help='image size', default=224)

    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--attn_dropout', type=float, default=0)
    parser.add_argument('--is_layer_norm', type=str2bool, default=False)
    parser.add_argument('--d_k', type=int, default=16)
    parser.add_argument('--d_v', type=int, default=16)

    parser.add_argument('--dropout', type=float, default=0.6)

    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--ldplr', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tddroprate', type=float, default=0.0)
    parser.add_argument('--droprate', type=float, default=0.0)
    parser.add_argument('--budroprate', type=float, default=0.0)
    parser.add_argument('--hid_feats', type=int, default=128)
    parser.add_argument('--out_feats', type=int, default=128)
    parser.add_argument('--diff_lr', type=str2bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--sem', type=float, default=0.3)

    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--intermediate_size', type=int, default=512)
    parser.add_argument('--v_hidden_size', type=int, default=512)
    parser.add_argument('--v_intermediate_size', type=int, default=512)
    parser.add_argument('--bi_num_attention_heads', type=int, default=8)
    parser.add_argument('--bi_hidden_size', type=int, default=512)
    parser.add_argument('--hidden_act', type=str, default="relu")
    parser.add_argument('--hidden_dropout_prob', type=float, default=0)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0)
    parser.add_argument('--v_attention_probs_dropout_prob', type=float, default=0)
    parser.add_argument('--v_hidden_dropout_prob', type=float, default=0)

    args = parser.parse_args()
    return args