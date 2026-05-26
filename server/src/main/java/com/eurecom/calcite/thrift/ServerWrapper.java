package com.eurecom.calcite.thrift;

import com.eurecom.calcite.thrift.CalciteServer.Processor;
import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TThreadPoolServer;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TTransportException;

public class ServerWrapper implements Runnable {
    private final ServerHandler handler;
    private final Processor processor;
    private TServer server;

    public ServerWrapper() {
        handler = new ServerHandler(server);
        processor = new Processor(handler);
    }

    private void startServer(Processor processor) throws TTransportException {
        TServerSocket socket = new TServerSocket(5555);
        server = new TThreadPoolServer(new TThreadPoolServer.Args(socket).processor(processor));
        System.out.println("Starting thrift server on port 5555");
        server.serve();
    }

    @Override
    public void run() {
        try {
            startServer(processor);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
