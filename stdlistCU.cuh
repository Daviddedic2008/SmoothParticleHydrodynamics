// list struct without any dynamic init to support requirements for usage in global device vars
#pragma once;
#include "stdinclude.cuh"

struct node {
	particlePlaceholder val;
	node* nextVal = nullptr;
};

struct linkedListCU {
	int size = 0;

	// void* used instead of node* to avoid so-called "dynamic allocation"...
	void* firstNode = nullptr;
	void* curNode = nullptr;
};

struct iteratorCU {
	node* curn = nullptr;
	node* prevn = nullptr;
};

inline __device__ iteratorCU getBeginningCU(const linkedListCU ls) {
	iteratorCU ret;
	ret.curn = ((node*)ls.firstNode);
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
	if (ls.size == 0) {
		ls.firstNode = new node;
		((node*)ls.firstNode)->val = p;
		ls.curNode = ls.firstNode;
		return getBeginningCU(ls);
	}
	((node*)ls.curNode)->nextVal = new node;

	ls.curNode = ((node*)ls.curNode)->nextVal;
	((node*)ls.curNode)->val = p;

	iteratorCU ret;
	ret.curn = ((node*)ls.curNode)->nextVal;
	ret.prevn = ((node*)ls.curNode);
	return ret;
}

inline __device__ void removeCU(linkedListCU& ls, const particlePlaceholder p) {
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
}

inline __device__ void removeImmediateCU(iteratorCU iter) {
	// iterator skips to next element, if there is one.

	if (iter.prevn == nullptr) {
		delete iter.curn;
		iter.curn = nullptr;
		return;
	}
	iter.prevn->nextVal = iter.curn->nextVal;
	delete iter.curn;
	iter.curn = iter.prevn->nextVal;
}

inline __device__ void addCU(linkedListCU& ls, const int index, const particlePlaceholder val) {
	iteratorCU iter = getBeginningCU(ls);

	for (int i = 0; i < index; i++) {
		advanceIteratorCU(iter);
	}

	node* tmp = iter.curn->nextVal;
	iter.curn->nextVal = new node;
	iter.curn->nextVal->val = val;

	if (iter.curn->nextVal == nullptr) {
		return;
	}
	iter.curn->nextVal->nextVal = tmp;
}

inline __device__ void addImmediateCU(iteratorCU& iter, const particlePlaceholder val) {
	node* tmp = iter.curn->nextVal;
	iter.curn->nextVal = new node;
	iter.curn->nextVal->val = val;

	if (tmp == nullptr) {
		return;
	}
	iter.curn->nextVal->nextVal = tmp;
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