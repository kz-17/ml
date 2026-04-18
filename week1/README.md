# week1
對話紀錄: https://gemini.google.com/share/728da21c98e7
### 鄰居變換 (2-opt Swap)
將路徑中的一段「反轉」  
Ex.  
原始： A -> B -> ... -> C -> D    
變換後： A -> C -> ... -> B -> D  
### 局部最優解 (Local Optimum)
爬山演算法最大的限制在於它非常「貪婪」。一旦周圍的鄰居都比目前的高度低，它就會停止  
**解決方案**： 如果結果不理想，通常會搭配 隨機重新開始 (Random Restart)，或是改用 模擬退火 (Simulated Annealing)，允許在一定機率下往低處走，以跳出局部陷阱  
