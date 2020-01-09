def data_generator():
    for i in range(10):
        yield (1,2)

datagen = data_generator()

print(len(list(datagen)))
