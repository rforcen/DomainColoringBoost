# DomainColoringBoost
python boost callable c++ multithread implementation of domain coloring algo. includes a z compiler so you can write z expressions directly in python as:

    from DomainColoring import DomainColoring

    import matplotlib.pyplot as plt
    from timeit import default_timer as time

    mf=2

    w, h = 1920*mf, 1080*mf

    lap=time()

    dc = DomainColoring(w, h, 'z^(5/c(1,2)) + sin(z) ^ z*c(1,8)')

    print('z compiler error for formula:', dc.formula, '->', dc.error)

    if not dc.error:

        print('domain coloring formula OK, image shape:', dc.w, 'x', dc.h, 'generation time:', time()-lap, 'secs')

        plt.imshow( dc.getimage_np() )
        plt.show()
