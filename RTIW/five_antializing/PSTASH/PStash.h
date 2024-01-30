#ifndef TPSTASH_H
#define TPSTASH_H
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

template<class T,int incr = 20>
class PStash {
  int quantity;
  int next;
  T storage[incr];
 public:
  PStash() : quantity(0), next(0) {}

  ~PStash();

  int add(T element);
  T operator[](int index) const;
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
      ++index;
      return *this;
    }

    iterator& operator++(int) {
      return operator++();
    }


    iterator& operator--() {  //prefix
      --index;
      return *this;
    }

    iterator& operator--(int) {  //postfix
      return operator--();
    }
    
    iterator& operator+=(int amount) {
      index += amount;
      return *this;
    }
    
    iterator& operator-=(int amount) {
      index -= amount;
      return *this;
    }

    iterator operator+(int amount) const {
      iterator ret(*this);
      ret += amount; // op+= does bounds check
      return ret;
    }

    T current () const {
      return p.storage[index];
    }

    T operator*() {
      return current();
    }
    
    T operator->() {
      require(p.storage[index] != 0,"PStash::iterator::operator->returns 0");
      return current();
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
int PStash<T,incr>::add(T element) {
  storage[next++] = element;
  return (next-1);
}


template<class T, int incr>
PStash<T, incr>::~PStash() {
  for(int i = 0; i < next; i++) {
    ///delete storage[i];
    storage[i] = 0;
  }
  //delete []storage;
}

template<class T,int incr>
T PStash<T,incr>::operator[](int index) const {
  //require(index >= 0,"negative index");
  if(index >= next) {
    return 0;
  }
  //require(storage[index] != 0, "No element at this record");
  return storage[index];
}



#endif