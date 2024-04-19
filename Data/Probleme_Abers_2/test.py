import pickle

with open("weight_Abers_pb2.pickle", "rb") as f:
    weight_Abers_pb2 = pickle.load(f)
    print(weight_Abers_pb2)
    print(sum(weight_Abers_pb2))
    print(len(weight_Abers_pb2))
with open("bilat_pairs_Abers_pb2.pickle", "rb") as f:
    bilat_pairs = pickle.load(f)
    print(bilat_pairs)

somme = 0
for elem in bilat_pairs:
    somme += weight_Abers_pb2[int(elem[0]) - 1]#
print(sum(weight_Abers_pb2[:-1]) - somme)