import heapq as hq

class TwoHeaps:

    def __init__(self):
        self.maxHeap = []  # contains the lower-half
        self.minHeap = []  # contains the upper-half

    # Returns the root of the specified queue - DONE
    def peek(self, heap):
        if (heap == 'max' and self.maxHeap != []): return self.maxHeap[0]
        elif (heap == 'min' and self.minHeap != []): return self.minHeap[0]
        else: raise Exception("The selected heap is empty")

    # Retrieves the root of the specified queue and then locally removes it - DONE
    def poll(self, heap):
        # For clarity, item is initialized at None and peek is perfomed inside 
        # the if-elif blocks
        item = None
        if (heap == "max"):
            item = -self.peek(heap)  # for the item to be returned as positive
            hq.heappop(self.maxHeap) 
        elif (heap == "min"):
            item = self.peek(heap) 
            hq.heappop(self.minHeap)
        return item

    # DONE
    def balanceHeaps(self):
        if (len(self.maxHeap) > len(self.minHeap) + 1):
            hq.heappush(self.minHeap, self.poll("max"))
        elif (len(self.maxHeap) < len(self.minHeap)):
            hq.heappush(self.maxHeap, -self.poll("min"))

    # For simplicity, the values inserted in the max heap are negative
    # When they're retrieved, the modulus operation is applied
    def insert(self, item):
        if (len(self.maxHeap) == 0 or -self.peek("max") >= item):
            hq.heappush(self.maxHeap, -item)
        else:
            hq.heappush(self.minHeap, item)
        self.balanceHeaps()

    def findMedian(self):
        if (len(self.maxHeap) == 0 and len(self.minHeap) == 0): raise Exception("Both heaps are empty!")
        elif ((len(self.minHeap) == 0) or (len(self.maxHeap) > len(self.minHeap))): return -self.peek("max")
        else: return (-self.peek("max") + self.peek("min"))/2 
    
    def printHeaps(self):
        print("MAX HEAP:\n")
        for i in self.maxHeap:
            print(i)
        
        print("\nMIN HEAP:")
        for i in self.minHeap:
            print(i)
            
        print("\nMAX HEAP: {}\nMIN HEAP: {}".format(len(self.maxHeap), len(self.minHeap)))