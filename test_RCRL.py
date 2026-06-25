import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import numpy as np
from utils import data_loader
from tqdm import tqdm
from utils.metrics import Evaluator
from PIL import Image
# from network.SemiModel2 import SemiModel
from network.RCRL import SemiModel
import matplotlib.pyplot as plt
import time





# === ADD: profiling dependencies (choose available ones automatically) ===
import csv
try:
    from fvcore.nn import FlopCountAnalysis
    _USE_FVCORE = True
except ImportError:
    _USE_FVCORE = False

try:
    from thop import profile
    _USE_THOP = True
except ImportError:
    _USE_THOP = False


start = time.time()


def test(test_loader, Eva_test, save_path, net):
    print("Strat validing!")

    # 创建保存注意力图的目录
    attention_save_dir = os.path.join(save_path, 'attention_maps')
    os.makedirs(attention_save_dir, exist_ok=True)
    
    net.train(False)
    net.eval()
    # === ADD: CUDA timing events ===
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times_per_img_ms = []  # 记录为“每张图”的毫秒

    
    for i, (A, B, mask, filename) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()

            torch.cuda.synchronize()
            starter.record()
            preds = net(A, B)
            ender.record()
            torch.cuda.synchronize()
            # 这一个batch的总毫秒，折算为“每张图”
            elapsed_ms = starter.elapsed_time(ender)
            times_per_img_ms.append(elapsed_ms / len(filename))            
            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy()

            for i in range(output.shape[0]):
                probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
                final_mask = probs_array * 255
                final_mask = final_mask.astype(np.uint8)
                final_savepath = save_path + filename[i] + ".png"
                im = Image.fromarray(final_mask)
                im.save(final_savepath)

            Eva_test.add_batch(target, pred)
    print("target.shape", target.shape)
    print("pred.shape", pred.shape)

    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA = Eva_test.OA()
    Kappa = Eva_test.Kappa()

    # print('[Test] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    print(
        "[Test] F1: %.4f, Precision:%.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f,IoU: %.4f"
        % (F1[1], Pre[1], Recall[1], OA[1], Kappa[1], IoU[1])
    )
    # print('F1-Score: {:.2f}\nPrecision: {:.2f}\nRecall: {:.2f}\nOA: {:.2f}\nKappa: {:.2f}\nIoU: {:.2f}\n}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100, IoU[1] * 100))
    print("F1-Score: Precision: Recall: OA: Kappa: IoU: ")
    # print('{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100,IoU[1] * 100))
    print(
        "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
            F1[1] * 100,
            Pre[1] * 100,
            Recall[1] * 100,
            OA[1] * 100,
            Kappa[1] * 100,
            IoU[1] * 100,
        )
    )
    # === ADD: report average test latency ===
    if len(times_per_img_ms) > 0:
        avg_ms = np.mean(times_per_img_ms)
        std_ms = np.std(times_per_img_ms)
        print(f"[Test] Inference time: {avg_ms:.2f} ± {std_ms:.2f} ms/img (bs={output.shape[0]})")
    # print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(F1[0] * 100, Pre[0] * 100, Recall[0] * 100, OA[0] * 100, Kappa[0] * 100,IoU[0] * 100))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=16, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=256, help="training dataset size"
    )
    parser.add_argument(
        "--gpu_id", type=str, default="1", help="train use gpu"
    )  # 修改这里！！！
    parser.add_argument(
        "--data_name",
        type=str,
        default="CDD",  # 修改这里！！！
        help="the test rgb images root",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SemiModel_CDD1",  # 修改这里！！！
        help="the test rgb images root",
    )

    parser.add_argument(
        "--save_path",
        type=str,     
        default="./test_result/RCRL-5-1/WHU/FPtt/",
    )  # 半监督影像保存路径！！！

    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == "0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("USE GPU 0")
    elif opt.gpu_id == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print("USE GPU 1")
    if opt.gpu_id == "2":
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print("USE GPU 2")
    if opt.gpu_id == "3":
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print("USE GPU 3")

    if opt.data_name == "LEVIR":
        opt.test_root = "/root/autodl-tmp/LEVIR_CD_256/test/"
    elif opt.data_name == "WHU":
        opt.test_root = (
            "/root/autodl-tmp/WHU-CD256/test/"
        )
    elif opt.data_name == "GoogleGZ":
        opt.test_root = "/root/autodl-tmp/GZ-CD256/test/"


    opt.save_path = opt.save_path + opt.data_name + "/" + opt.model_name + "/"
    test_loader = data_loader.get_test_loader(
        opt.test_root,
        opt.batchsize,
        opt.trainsize,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    Eva_test = Evaluator(num_class=2)
    if opt.model_name == "HANet_v2":
        model = HANet_v2().cuda()
    elif opt.model_name == "SemiModel_CDD1":
        model = SemiModel().cuda()
    elif opt.model_name == "CRCL_LEVIR1":
        model = SemiModel().cuda()
    elif opt.model_name == "SemiModel2_GoogleGZ1":
        model = SemiModel().cuda()
    elif opt.model_name == "CRCL_WHU1":
        model = SemiModel().cuda()
    elif opt.model_name == "CRCL_GoogleGZ1":
        model = SemiModel().cuda()

    save_path = "./output/RCRL-SemiCD/WHU-5-1/"  # 半监督SemiCD模型路径！！

    
    save_path = save_path + opt.data_name + "/" + opt.model_name

    opt.load = save_path + "_train1_" + "_best_student_iou.pth"
    # opt.load = save_path + "_train1_" + "_best_teacher_iou.pth" # 教师模型
    # opt.load ='./output/LEVIR-5%/SemiModel_noema04_best_teacher_iou.pth'
    if opt.load is not None:
        print("load model from ", opt.load)
        checkpoint_stud = torch.load(opt.load)
        model.load_state_dict(checkpoint_stud["best_student_net "])
        # model.load_state_dict(checkpoint_stud) # 教师模型

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        # === ADD: Params & FLOPs (run once) ===
    model.eval()
    device = next(model.parameters()).device
    H = W = opt.trainsize  # 与测试尺寸一致
    dummy_A = torch.randn(1, 3, H, W, device=device)
    dummy_B = torch.randn(1, 3, H, W, device=device)

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Profile] Params: {total_params/1e6:.2f} M")

    # FLOPs (prefer fvcore; fallback to thop)
    gflops = None
    if _USE_FVCORE:
        flops = FlopCountAnalysis(model, (dummy_A, dummy_B))
        gflops = flops.total() / 1e9
        print(f"[Profile] FLOPs: {gflops:.2f} GFLOPs")
    elif _USE_THOP:
        macs, _ = profile(model, inputs=(dummy_A, dummy_B), verbose=False)
        gflops = (macs * 2) / 1e9 
        print(f"[Profile] MACs: {macs/1e9:.2f} GMacs (~{gflops:.2f} GFLOPs)")
    else:
        print("[Profile] Skip FLOPs: neither fvcore nor thop is available.")


    
    test(test_loader, Eva_test, opt.save_path, model)

end = time.time()
print("程序测试test的时间为:", end - start)
