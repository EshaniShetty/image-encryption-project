from chaos import SkewTentMap

chaos = SkewTentMap(x0=0.45, p=0.35)

sequence = chaos.generate_sequence(20)

print("\nChaotic Sequence:\n")

for i, value in enumerate(sequence):
    print(f"{i+1:2d} --> {value:.10f}")