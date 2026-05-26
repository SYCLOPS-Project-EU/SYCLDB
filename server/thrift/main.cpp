#include <iostream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

// using namespace

int main() {
  std::shared_ptr<TTransport> socket(new TSocket("localhost", 5555));
  std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  CalciteServerClient client(protocol);

  try {
    transport->open();

    std::string sql
        = "select d_year,c_nation, COUNT(*) as profit\
            from lineorder,\
                supplier, customer, part,\
        ddate where lo_custkey\
        = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate\
        = d_datekey and c_region = 1 and s_region\
        = 1 and (p_mfgr = 0 or p_mfgr = 1) group by d_year,\
                        c_nation;";
    PlanResult result;
    client.parse(result, sql);

    std::cout << "Result: " << result << std::endl;

    transport->close();
  } catch (TTransportException& e) {
    std::cerr << "Transport exception: " << e.what() << std::endl;
  } catch (TException& e) {
    std::cerr << "Thrift exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception" << std::endl;
  }

  return 0;
}