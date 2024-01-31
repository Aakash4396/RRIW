#ifndef TPSTASH_H
#define TPSTASH_H
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

__host__ __device__ void require(bool requirement, const char* msg = "Requirement failed") {
    if(!requirement) {
        printf(msg);
        printf("\n");
        //exit(1);
    }
}

template<class T,int incr = 1>
class PStash {
    int quantity;
    int next;
    T** storage;
    __host__ __device__ void inflate(int increase = incr);
public:
    __host__ __device__ PStash() : quantity(0), storage(0), next(0) {}

    __host__ __device__ ~PStash();

    __host__ __device__ int add(T* element);
    __host__ __device__ T* operator[](int index) const;
    __host__ __device__ int count() const { return next; }
    
    class iterator {
        PStash& p; 
        int index;
    public:
        __host__ __device__ iterator(PStash& ob):p(ob), index(0) {}
        __host__ __device__ iterator(PStash& ob, bool b):p(ob), index(ob.next) {}

        __host__ __device__ iterator(const iterator& itr2):p(itr2.p), index(itr2.index) {}
        
        __host__ __device__ iterator& operator=(const iterator& itr2) { 
            p = itr2.p;
            index = itr2.index;
            return *this;
        }

        __host__ __device__ iterator& operator++() {  //prefix
            require(++index <= p.next,"Index out of bound");
            return *this;
        }

        __host__ __device__ iterator& operator++(int) {
            return operator++();
        }


        __host__ __device__ iterator& operator--() {  //prefix
            require(--index >= 0,"Index out of bound");
            return *this;
        }

        __host__ __device__ iterator& operator--(int) {  //postfix
            return operator--();
        }
        
        __host__ __device__ iterator& operator+=(int amount) {
            require(index+amount < p.next && index+amount >= 0,"Index out of bound");
            index += amount;
            return *this;
        }
        
        __host__ __device__ iterator& operator-=(int amount) {
            require(index-amount < p.next && index - amount >= 0,"Index out of bound");
            index -= amount;
            return *this;
        }

        __host__ __device__ iterator operator+(int amount) const {
            iterator ret(*this);
            ret += amount; // op+= does bounds check
            return ret;
        }

        __host__ __device__ T* current () const {
            return p.storage[index];
        }

        __host__ __device__ T* operator*() {
            return current();
        }
        
        __host__ __device__ T* operator->() {
            require(p.storage[index] != 0,"PStash::iterator::operator->returns 0");
            return current();
        }

        __host__ __device__ bool operator==(const iterator& rv) const {
            return index == rv.index;
        }

        __host__ __device__ bool operator!=(const iterator& rv) const {
            return index != rv.index;
        }
    };

    __host__ __device__ iterator begin() {
        return iterator(*this);
    }
    
    __host__ __device__ iterator end() {
        return iterator(*this,true);
    }
};



template<class T, int incr>
__host__ __device__ int PStash<T,incr>::add(T* element) {
    if(next >= quantity)
        inflate(incr);
    storage[next++] = element;
    return (next-1);
}


template<class T, int incr>
__host__ __device__ PStash<T, incr>::~PStash() {
    for(int i = 0; i < next; i++) {
        delete storage[i];
        storage[i] = 0;
    }
    delete []storage;
}

template<class T,int incr>
__host__ __device__ T* PStash<T,incr>::operator[](int index) const {
    require(index >= 0,"negative index");
    if(index >= next) {
        return 0;
    }
    require(storage[index] != 0, "No element at this record");
    return storage[index];
}

template<class T, int incr>
__host__ __device__ void PStash<T,incr>::inflate(int increase) {
    const int psz = sizeof(T*);
    T** st = new T*[quantity + increase];
    memset(st,0,(quantity + increase) * psz);
    memcpy(st,storage,(quantity) * psz);
    quantity += increase;
    delete []storage;
    storage = st;
}


#endif
