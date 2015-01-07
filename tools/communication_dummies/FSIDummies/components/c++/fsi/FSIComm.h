#ifndef FSI_FSICOMM_H_
#define FSI_FSICOMM_H_ 

//
// ASCoDT - Advanced Scientific Computing Development Toolkit
//
// This file was generated by ASCoDT's simplified SIDL compiler.
//
// Authors: Tobias Weinzierl, Atanas Atanasov   
//

#include <iostream>
#include <string>



namespace fsi { 

     class FSIComm;
}

class fsi::FSIComm {
  public:
    virtual ~FSIComm(){}
     virtual void endDataTransfer(int& ack)=0;
     virtual void endDataTransferParallel(int& ack)=0;
     virtual void startDataTransfer()=0;
     virtual void startDataTransferParallel()=0;
     virtual void transferCoordinates(const int* coordId, const int coordId_len,const int* offsets, const int offsets_len,const std::string* hosts, const int hosts_len)=0;
     virtual void transferCoordinatesParallel(const int* coordId, const int coordId_len,const int* offsets, const int offsets_len,const std::string* hosts, const int hosts_len)=0;


};

#endif
