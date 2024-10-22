import torch,time
import torch_tools
if __name__=="__main__":
    A = 2048
    B = 48
    K = 2
    other = 48
    a = torch.randn(A,B,other).cuda(1).half()
    b = [A, K, B, other]
    index = torch.randint(0,K,[A,B]).cuda(1)
    # shape_a = torch.tensor(a.size(),dtype=torch.int).cuda()
    # shape_index = torch.tensor(index.size(),dtype=torch.int).cuda()
    # shape_b = torch.tensor(b.size(),dtype=torch.int).cuda()
    print(a.size(),b,index.size())
    start = time.time()
    res = torch_tools.mapping(a,index,b)
    end = time.time()
    print(index)
    print(res)
    print(end - start)

    x = torch.rand(A,K).cuda(1).half()
    y = [A, B, other]
    print(x.size(),y,index.size())
    start = time.time()
    res = torch_tools.mapping(x,index,y)
    end = time.time()
    print(x)
    print(index)
    print(res)
    print(end - start)
