from CustomDatasets.ClassSampledMNIST import SampledMNIST

dataset = SampledMNIST(10_000)

print(dataset.__getitem__(0))
