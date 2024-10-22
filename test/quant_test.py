import torch,time
import torch_tools

if __name__=="__main__":
    a = torch.randn(40,40,40,40).cuda().half()
    exp = torch.ones_like(a)
    max_number = 2**(3)*(2-2**(-3))
    emax = 3
    emin = -4
    man = 5
    data = torch.cuda.memory_allocated()
    print("original data:{:10.2f}m".format(data/1024/1024))

    start = time.time()

    offset = exp


    shift = 2 ** (-offset)
    i = a * shift

    emin = emin
    # 指数位能表示的最大值
    emax = emax  # number.of_emax

    esbn = 2 ** (emin + 1)
    lsbn = 2 ** (emax)
    mval = 2 ** (man)
    # 这个数的最大值
    rlim = max_number

    sgn = torch.sign(i)

    i = torch.abs(i)

    e = torch.floor(torch.log2(i + 1e-60))


    ie = i * 2 ** (-e)
    me = 2 ** (e)

    f = torch.where(i < esbn, ie, ie - 1)

    r = torch.rand_like(f)
    f.mul_(mval).add_(r).floor_()

    clipped = f.clamp_(0, mval)
    clipped.div_(mval).mul_(me)
    k = torch.where(i < esbn, clipped, me + clipped)
    k.clamp_(-rlim, rlim)
    out = sgn * k * 2 ** (offset)

    end = time.time()
    print("original comsume:{:8.5f}s".format(end - start))


    ori_method = torch.cuda.memory_allocated()
    print("original method:{:10.2f}m".format((ori_method - data)/1024/1024))

    start = time.time()
    res = torch_tools.BFPquant(a, exp, emin, emax, man,True)
    end = time.time()
    print("refactor comsume:{:8.5f}s".format(end - start))

    
    print("refactor method:{:10.2f}m".format((torch.cuda.memory_allocated() - ori_method)/1024/1024))

    print("original max diff:", (a - out).abs().max())
    print("refactor max diff:", (a - res).abs().max())