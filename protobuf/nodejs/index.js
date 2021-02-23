var ProtoBuf = require("protobufjs");

ProtoBuf.load("./user.proto", function (err, root) {
    User = root.lookupType("User");
    Product = root.lookupType("Product");

    var product1 = {
        id: 1,
        name: "product1",
        price: 1200,
    };
    var product2 = {
        id: 2,
        name: "product2",
        price: 3400,
    };

    var user1 = {
        name: "user1",
        age: 24,
        products: [product1, product2],
    };

    var errMsg = User.verify(user1);
    if (errMsg) throw Error(errMsg);

    var user1_obj = User.create(user1);
    var data = User.encode(user1_obj).finish();

    let obj = User.decode(data);
    console.log(obj);
});

