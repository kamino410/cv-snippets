import user_pb2

u1 = user_pb2.User()
u1.name = "user1"
u1.age = 24

p1 = u1.products.add()
p1.id = 1
p1.name = "product1"
p1.price = 1300

p2 = user_pb2.Product()
p2.id = 2
p2.name = "product2"
p2.price = 5100
u1.products.append(p2)

data = u1.SerializeToString()

user = user_pb2.User()
user.ParseFromString(data)
print(user)
