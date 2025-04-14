# Group generators for Z_15. 
generators = {10}
modulo = 10
cyclic_subgroups = []

MAX_ITERATIONS = 100

for generator in generators:
    subgroup = []
    
    for multiple in range(1, modulo + 1):
        computed = (generator * multiple) % modulo
        subgroup.append(computed)

        if computed == 0:
            break
        
    cyclic_subgroups.append((generator, subgroup))


for generator, subgroup in cyclic_subgroups:
    print(f"<{generator}> = {subgroup}\n")


