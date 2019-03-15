#include <vector>
#include <map>
#include <iostream>
#include <windows.h>
#include <sstream>

#include "optitrack_client.hpp"

namespace NatNetImpl {
std::vector<sNatNetDiscoveredServer> discoveredServers;
std::map<int, RigidBody> bodies;

void NATNET_CALLCONV ServerDiscoveredCallback(
    const sNatNetDiscoveredServer* pDiscoveredServer, void* pUserContext) {
  discoveredServers.push_back(*pDiscoveredServer);
}

void NATNET_CALLCONV DataHandler(sFrameOfMocapData* data, void* pUserData) {
  for (auto body : data->RigidBodies) {
    RigidBody *tar = &bodies[body.ID];
    tar->x = body.x;
    tar->y = body.y;
    tar->z = body.z;
    tar->qx = body.qx;
    tar->qy = body.qy;
    tar->qz = body.qz;
    tar->qw = body.qw;
    tar->error = body.MeanError;
    tar->valid = body.params & 0x01;
  }
}
}  // namespace NatNetImpl

using namespace NatNetImpl;

OptiTrackClient::OptiTrackClient() {}

OptiTrackClient::~OptiTrackClient() {
  if (client) {
    client->Disconnect();
    delete client;
  }
}

bool OptiTrackClient::connect() {
  NatNetDiscoveryHandle discovery;
  NatNet_CreateAsyncServerDiscovery(&discovery, ServerDiscoveredCallback);
  for (int i = 0; i < 20; i++) {
    if (discoveredServers.size()) break;
    Sleep(500);
  }
  NatNet_FreeAsyncServerDiscovery(discovery);

  if (discoveredServers.size() == 0) {
    std::cout << "OptiTrack Server is not found" << std::endl;
    return false;
  }

  server = discoveredServers[0];

  // print server info
  std::cout << server.serverDescription.szHostApp << " "
            << server.serverDescription.HostAppVersion[0] << "."
            << server.serverDescription.HostAppVersion[1] << " "
            << server.serverAddress << std::endl;
  if (server.serverDescription.bConnectionInfoValid == false) {
    std::cout << "(WARNING: Legacy server, could not autodetect settings. "
                 "Auto-connect may not work reliably.)"
              << std::endl;
  }

  // set connect parameters
  sNatNetClientConnectParams connectParams;
  if (server.serverDescription.bConnectionInfoValid) {
    // Build the connection parameters.
    std::stringstream ss;
    ss << server.serverDescription.ConnectionMulticastAddress[0] << "."
       << server.serverDescription.ConnectionMulticastAddress[1] << "."
       << server.serverDescription.ConnectionMulticastAddress[2] << "."
       << server.serverDescription.ConnectionMulticastAddress[3];

    connectParams.connectionType = server.serverDescription.ConnectionMulticast
                                       ? ConnectionType_Multicast
                                       : ConnectionType_Unicast;
    connectParams.serverCommandPort = server.serverCommandPort;
    connectParams.serverDataPort = server.serverDescription.ConnectionDataPort;
    connectParams.serverAddress = server.serverAddress;
    connectParams.localAddress = server.localAddress;
    connectParams.multicastAddress = ss.str().c_str();
  } else {
    // We're missing some info because it's a legacy server.
    // Guess the defaults and make a best effort attempt to connect.
    connectParams.connectionType = ConnectionType_Multicast;
    connectParams.serverCommandPort = server.serverCommandPort;
    connectParams.serverDataPort = 0;
    connectParams.serverAddress = server.serverAddress;
    connectParams.localAddress = server.localAddress;
    connectParams.multicastAddress = NULL;
  }

  client = new NatNetClient();
  if (client->Connect(connectParams) != ErrorCode_OK) {
    std::cout << "Failed to connect server." << std::endl;
    return false;
  }

  sDataDescriptions* pDataDefs;
  if (client->GetDataDescriptionList(&pDataDefs) != ErrorCode_OK) {
    std::cout << "Failed to retrieve data descriptions." << std::endl;
    return false;
  }

  for (auto desc : pDataDefs->arrDataDescriptions) {
    if (desc.type == Descriptor_RigidBody) {
      sRigidBodyDescription* pRB = desc.Data.RigidBodyDescription;
      std::cout << "Rigid Body" << std::endl;
      std::cout << "  Name : " << pRB->szName << std::endl;
      std::cout << "  ID : " << pRB->ID << std::endl;
      std::cout << "  Parent ID : " << pRB->parentID << std::endl;
      std::cout << "  Parent Offset : " << pRB->offsetx << ", " << pRB->offsety
                << ", " << pRB->offsetz << std::endl;
    }
  }

  client->SetFrameReceivedCallback(DataHandler, client);

  return true;
}

RigidBody OptiTrackClient::getBody(int id) { return bodies[id]; }
