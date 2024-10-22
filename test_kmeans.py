import torch
import torch_tools,time

if __name__=="__main__":

    a = torch.randn(1000,48,48).cuda(2)
    b = torch.randn(1000,48,48).cuda(2)

    res = torch_tools._C.add(a,b)

    print(res)