#include <user.pb.h>

#include <iostream>
using namespace std;

int main(int argc, char const* argv[]) {
  User u1;
  u1.set_name("user1");
  u1.set_age(24);

  Product p1;
  p1.set_id(1);
  p1.set_name("product1");
  p1.set_price(1200);

  auto new_p = u1.add_products();
  new_p->CopyFrom(p1);

  auto p2 = u1.add_products();
  p2->set_id(2);
  p2->set_name("product2");
  p2->set_price(3100);

  string serialized_str;
  u1.SerializeToString(&serialized_str);

  User user;
  user.ParseFromString(serialized_str);
  cout << user.products(1).price() << std::endl;

  return 0;
}

