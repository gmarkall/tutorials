
import builtins
profile = getattr(builtins, "profile", lambda x: x)

size = 10000

@profile
def f(size):
    l = list(range(size))
    sum(l)
    s = []
    for idx in range(size):
        s.append(sum(l[idx:]))

if __name__ == "__main__":
    f(size)
    
