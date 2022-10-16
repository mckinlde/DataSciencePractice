def func(*args):
    print('func(',*args,')')
    def inside_func(x):
        print('inside')

        def inside_inside(*args):
            print('inin(',*args,')')
            return x(*args)
        return inside_inside
    return inside_func

@func(1,2,3)
def bar():
    print('bar')

print('outside')