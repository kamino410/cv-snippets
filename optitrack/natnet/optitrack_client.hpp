#pragma once
#include <NatNetCAPI.h>
#include <NatNetClient.h>

struct RigidBody {
  float x, y, z;
  float qx, qy, qz, qw;
  float error;
  bool valid;
};

class OptiTrackClient {
  sNatNetDiscoveredServer server;
  NatNetClient* client;

 public:
  OptiTrackClient();
  ~OptiTrackClient();

  bool connect();
  RigidBody getBody(int id);
};
