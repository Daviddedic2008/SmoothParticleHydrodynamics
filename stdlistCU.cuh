// list struct without any dynamic init to support requirements for usage in global device vars
#pragma once;
#include "stdinclude.cuh"

struct node {
	particlePlaceholder val;
	node* nextVal = nullptr;
};

struct linkedListCU {
	int size = 0;
	int* mutex = nullptr;

	// void* used instead of node* to avoid so-called "dynamic allocation"...
	void* firstNode = nullptr;
	void* curNode = nullptr;
};

struct iteratorCU {
	node* curn = nullptr;
	node* prevn = nullptr;
};

inline __device__ void takeList(const linkedListCU ls) {
	while (atomicCAS(ls.mutex, 0, 1) != 0) {
		;
	}
	atomicExch(ls.mutex, 0);
}

inline __device__ void releaseList(const linkedListCU ls) {
	atomicExch(ls.mutex, 1);
}

inline __device__ void initList(linkedListCU& ls) {
	ls.mutex = new int;
	*ls.mutex = 0;
}

inline __device__ iteratorCU getBeginningCU(const linkedListCU ls) {
	takeList(ls);
	iteratorCU ret;
	ret.curn = ((node*)ls.firstNode);
	releaseList(ls);

	return ret;
}

inline __device__ void advanceIteratorCU(iteratorCU& iter) {
	iter.prevn = iter.curn;
	iter.curn = iter.curn->nextVal;
}

inline __device__ particlePlaceholder getIteratorValueCU(const iteratorCU iter) {
	// too much function call overhead to use
	return iter.curn->val;
}

inline __device__ iteratorCU push_backLinkedCU(linkedListCU& ls, const particlePlaceholder p) {
	takeList(ls);
	if (ls.size == 0) {
		ls.firstNode = new node;
		((node*)ls.firstNode)->val = p;
		ls.curNode = ls.firstNode;
		releaseList(ls);
		return getBeginningCU(ls);
	}
	((node*)ls.curNode)->nextVal = new node;

	ls.curNode = ((node*)ls.curNode)->nextVal;
	((node*)ls.curNode)->val = p;

	iteratorCU ret;
	ret.curn = ((node*)ls.curNode)->nextVal;
	ret.prevn = ((node*)ls.curNode);
	releaseList(ls);
	return ret;
}

inline __device__ void removeCU(linkedListCU& ls, const particlePlaceholder p) {
	takeList(ls);
	node* curn = ((node*)ls.firstNode);
	node* prevNode = nullptr;
	while (true) {
		if (particlePlaceholderEquals(curn->val, p)) {
			if (prevNode == nullptr) {
				const node* tmp = ((node*)ls.firstNode);
				ls.firstNode = ((node*)ls.firstNode)->nextVal;
				delete tmp;
				break;
			}
			prevNode->nextVal = curn->nextVal;
			delete curn;
			break;
		}

		if (curn->nextVal->nextVal == nullptr) {
			break;
		}
		prevNode = curn;
		curn = curn->nextVal;
	}
	releaseList(ls);
}

inline __device__ void removeImmediateCU(iteratorCU& iter, linkedListCU ls) {
	// iterator skips to next element, if there is one.

	takeList(ls);
	if (iter.curn == nullptr) { releaseList(ls); return; }

	if (iter.prevn == nullptr) {
		delete iter.curn;
		iter.curn = nullptr;
		releaseList(ls);
		return;
	}
	iter.prevn->nextVal = iter.curn->nextVal;
	delete iter.curn;
	iter.curn = iter.prevn->nextVal;
	releaseList(ls);
}

inline __device__ void addCU(linkedListCU& ls, const int index, const particlePlaceholder val) {

	takeList(ls);
	iteratorCU iter = getBeginningCU(ls);

	for (int i = 0; i < index; i++) {
		advanceIteratorCU(iter);
	}

	node* tmp = iter.curn->nextVal;
	iter.curn->nextVal = new node;
	iter.curn->nextVal->val = val;

	if (iter.curn->nextVal == nullptr) {
		releaseList(ls);
		return;
	}

	iter.curn->nextVal->nextVal = tmp;
	releaseList(ls);
}

inline __device__ void addImmediateCU(iteratorCU& iter, const particlePlaceholder val, linkedListCU ls) {

	takeList(ls);
	node* tmp = iter.curn->nextVal;
	iter.curn->nextVal = new node;
	iter.curn->nextVal->val = val;

	if (tmp == nullptr) {
		releaseList(ls);
		return;
	}
	iter.curn->nextVal->nextVal = tmp;
	releaseList(ls);
}

struct vectorCU {
	particlePlaceholder* data = nullptr;
	int size = 0, currentIndex = 0;
};

inline __device__ void push_backVectorCU(vectorCU& vec, const particlePlaceholder p) {
	if (vec.currentIndex >= vec.size) {
		particlePlaceholder* tmp = new particlePlaceholder[vec.size * 2 + 1];
		for (int i = 0; i < vec.size; i++) {
			tmp[i] = vec.data[i];
		}
		delete vec.data;
		vec.data = tmp;
	}

	vec.currentIndex++;
	vec.data[vec.size] = p;
	vec.size = vec.size * 2 + 1;
}

inline __device__ void freeCU(vectorCU& vec) {
	delete vec.data;
	vec.size = 0;
	vec.currentIndex = 0;
}

inline __device__ particlePlaceholder getElementCU(vectorCU vec, const int index) {
	return vec.data[index];
}

inline __device__ particlePlaceholder setElementCU(vectorCU vec, const int index, const particlePlaceholder p) {
	vec.data[index] = p;
}