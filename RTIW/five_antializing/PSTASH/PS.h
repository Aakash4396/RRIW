#ifndef TPSTASH_H
#define TPSTASH_H
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

inline void require(bool requirement, const std::string& msg = "Requirement failed") {
  using namespace std;
  if(!requirement) {
    fputs(msg.c_str(),stderr);
    fputs("\n",stderr);
    exit(1);
  }
}

template<class T,int incr = 20>
class PStash {
  int quantity;
  int next;
  T** storage;
  void inflate(int increase = incr);
 public:
  PStash() : quantity(0), storage(0), next(0) {}

  ~PStash();

  int add(T* element);
  T* operator[](int index) const;
  T* remove(int index);
  int count() const { return next; }
  
  class iterator {
    PStash& p; 
    int index;
   public:
    iterator(PStash& ob):p(ob), index(0) {}
    iterator(PStash& ob, bool b):p(ob), index(ob.next) {}

    iterator(const iterator& itr2):p(itr2.p), index(itr2.index) {}
    
    iterator& operator=(const iterator& itr2) { 
      p = itr2.p;
      index = itr2.index;
      return *this;
    }

    iterator& operator++() {  //prefix
      require(++index <= p.next,"Index out of bound");
      return *this;
    }

    iterator& operator++(int) {
      return operator++();
    }


    iterator& operator--() {  //prefix
      require(--index >= 0,"Index out of bound");
      return *this;
    }

    iterator& operator--(int) {  //postfix
      return operator--();
    }
    
    iterator& operator+=(int amount) {
      require(index+amount < p.next && index+amount >= 0,"Index out of bound");
      index += amount;
      return *this;
    }
    
    iterator& operator-=(int amount) {
      require(index-amount < p.next && index - amount >= 0,"Index out of bound");
      index -= amount;
      return *this;
    }

    iterator operator+(int amount) const {
      iterator ret(*this);
      ret += amount; // op+= does bounds check
      return ret;
    }

    T* current () const {
      return p.storage[index];
    }

    T* operator*() {
      return current();
    }
    
    T* operator->() {
      require(p.storage[index] != 0,"PStash::iterator::operator->returns 0");
      return current();
    }

    // Remove the current element:
    T* remove(){
      return p.remove(index);
    }

    bool operator==(const iterator& rv) const {
      return index == rv.index;
    }

    bool operator!=(const iterator& rv) const {
      return index != rv.index;
    }
  };

  iterator begin() {
    return iterator(*this);
  }
  
  iterator end() {
    return iterator(*this,true);
  }
};



template<class T, int incr>
int PStash<T,incr>::add(T* element) {
  if(next >= quantity)
    inflate(incr);
  storage[next++] = element;
  return (next-1);
}


template<class T, int incr>
PStash<T, incr>::~PStash() {
  for(int i = 0; i < next; i++) {
    delete storage[i];
    storage[i] = 0;
  }
  delete []storage;
}

template<class T,int incr>
T* PStash<T,incr>::operator[](int index) const {
  require(index >= 0,"negative index");
  if(index >= next) {
    return 0;
  }
  require(storage[index] != 0, "No element at this record");
  return storage[index];
}

template<class T, int incr>
T* PStash<T,incr>::remove(int index) {
  T* v = operator[](index);
  if(v != 0)
    storage[index] = 0;
  return v;
}

template<class T, int incr>
void PStash<T,incr>::inflate(int increase) {
  const int psz = sizeof(T*);
  T** st = new T*[quantity + increase];
  memset(st,0,(quantity + increase) * psz);
  memcpy(st,storage,(quantity) * psz);
  quantity += increase;
  delete []storage;
  storage = st;
}


#endif
