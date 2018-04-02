package nowcoder;

import java.util.*;

public class Solution {

	/**
	 * 
	 * 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
	 * 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
	 * 
	 * @param target
	 * @param array
	 * @return
	 */
	public boolean Find(int target, int[][] array) {

		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				if (target == array[i][j])
					return true;
			}
		}

		return false;
	}

	public boolean Find2(int target, int[][] array) {

		int len = array.length - 1;
		int i = 0;

		while (len > -1 && i < array[0].length) {
			if (array[len][i] > target) {
				len--;
			} else if (array[len][i] < target) {
				i++;
			} else
				return true;
		}

		return false;
	}

	/**
	 * 
	 * 请实现一个函数，将一个字符串中的空格替换成“%20”
	 * 
	 * 例: Input: We Are Happy Output: We%20Are%20Happy。
	 * 
	 * @param str
	 * @return
	 */
	public String replaceSpace(StringBuffer str) {

		int i = str.indexOf(" ", 0);

		while (i < str.length() - 1 && i != -1) {
			str.replace(i, i + 1, "%20");
			i = str.indexOf(" ", i);
		}

		String s = str.toString();
		if (i == str.length() - 1 && i != -1)
			s = s.trim() + "%20";

		return s;
	}

	/**
	 * 
	 * 输入一个链表，从尾到头打印链表每个节点的值
	 * 
	 * @param listNode
	 * @return
	 */
	public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
		ArrayList<Integer> list = new ArrayList<Integer>();

		if (listNode != null) {
			list = printListFromTailToHead(listNode.next);
			list.add(listNode.val);
		}

		return list;
	}

	/**
	 * 
	 * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
	 * 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
	 * 
	 * @param pre
	 * @param in
	 * @return
	 */

	static int i = 0;

	public TreeNode reConstructBinaryTree(int[] pre, int[] in) {

		int j = 0;

		TreeNode root = null;

		if (pre.length == 0 || in.length == 0)
			return root;

		for (int inx = 0; inx < in.length; inx++) {
			if (in[inx] == pre[i]) {
				j = inx;
				break;
			}
		}
		root = new TreeNode(pre[i++]);
		root.left = reBinaryTree(pre, in, 0, j - 1);
		root.right = reBinaryTree(pre, in, j + 1, pre.length - 1);

		i = 0;
		return root;
	}

	// 递归创建树
	private TreeNode reBinaryTree(int[] pre, int[] in, int start, int end) {
		if (start > end)
			return null;

		int j = 0;
		for (int inx = start; inx <= end; inx++) {
			if (in[inx] == pre[i]) {
				j = inx;
				break;
			}
		}

		TreeNode node = new TreeNode(pre[i++]);
		node.left = reBinaryTree(pre, in, start, j - 1);
		node.right = reBinaryTree(pre, in, j + 1, end);

		return node;
	}

	/*
	 * 栈方式实现 public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
	 * Stack<TreeNode> lTree = new Stack<TreeNode>(); Stack<Integer> inx = new
	 * Stack<Integer>();
	 * 
	 * int i = 0;
	 * 
	 * while(i < pre.length){
	 * 
	 * } }
	 */

	/**
	 * 
	 * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
	 * 
	 * @param node
	 */
	Stack<Integer> stack1 = new Stack<Integer>();
	Stack<Integer> stack2 = new Stack<Integer>();

	public void push(int node) {
		stack1.push(node);
	}

	public int pop() {

		if (stack2.isEmpty() && !stack1.isEmpty()) {
			while (!stack1.isEmpty())
				stack2.push(stack1.pop());
		}

		if (!stack2.isEmpty())
			return stack2.pop();

		return 0;
	}

	/**
	 * 
	 * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
	 * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
	 * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
	 * 
	 * @param array
	 * @return
	 */
	// 二分法变种
	public int minNumberInRotateArray(int[] array) {

		if (array.length == 0)
			return 0;

		int low = 0, height = array.length - 1, mid = 0;

		while (low + 1 != height) {
			mid = (low + height) / 2;

			if (array[low] > array[mid])
				height = mid;
			else if (array[low] < array[mid])
				low = mid;
			else
				low++;
		}

		return array[height];
	}

	/**
	 * 
	 * 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。 n<=39
	 * 
	 * @param n
	 * @return
	 */
	public int Fibonacci(int n) {
		int pre = 0, last = 1, now = 0;

		if (n == 1)
			return 1;

		for (int i = 2; i <= n; i++) {
			now = pre + last;
			pre = last;
			last = now;
		}

		return now;
	}

	/**
	 * 
	 * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
	 * 
	 * @param target
	 * @return
	 */
	// F[n] = F[n - 1] + F[n - 2]
	public int JumpFloor(int target) {
		int first = 0, second = 1, num = 0;

		for (int i = 1; i <= target; i++) {
			num = first + second;
			first = second;
			second = num;
		}

		return num;
	}

	/**
	 * 
	 * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
	 * 
	 * @param target
	 * @return
	 */
	public int JumpFloorII(int target) {

		if (target <= 0)
			return 0;

		int num = 1;

		while (target > 1) {
			num *= 2;
			target--;
		}

		return num;
	}

	/**
	 * 
	 * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
	 * 
	 * @param target
	 * @return
	 */
	public int RectCover(int target) {

		if (target <= 0)
			return 0;

		int f1 = 1, f2 = 1, sum = 1;

		for (int i = 2; i <= target; i++) {
			sum = f1 + f2;
			f1 = f2;
			f2 = sum;
		}

		return sum;
	}

	/**
	 * 
	 * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
	 * 
	 * 1符号位 + 31数字位
	 * 
	 * @param n
	 * @return
	 */
	public int NumberOf1(int n) {
		int count = 0;
		while (n != 0) {
			count++;
			n = n & (n - 1);
		}
		return count;
	}

	/**
	 * 
	 * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
	 * 
	 * @param base
	 * @param exponent
	 * @return
	 */
	public double Power(double base, int exponent) {

		return Math.pow(base, exponent);
	}

	/**
	 * 
	 * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序， 使得所有的奇数位于数组的前半部分，所有的偶数位于位于数
	 * 组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位 置不变。
	 * 
	 * @param array
	 */
	public void reOrderArray(int[] array) {

		int num = 0, inx = -1;

		for (int i = 0; i < array.length; i++) {
			if (inx == -1 && array[i] % 2 == 0)
				inx = i;

			if (inx != -1 && array[i] % 2 != 0) {
				num = array[i];
				for (int j = i; j > inx; j--)
					array[j] = array[j - 1];
				array[inx++] = num;
			}
		}
	}

	/**
	 * 
	 * 输入一个链表，输出该链表中倒数第k个结点。
	 * 
	 * @param head
	 * @param k
	 * @return
	 */
	public ListNode FindKthToTail(ListNode head, int k) {

		ListNode node = head;
		int size = 0, i = 0;

		while (node != null) {
			node = node.next;
			size++;
		}

		if (k > size)
			return null;

		while (i < size - k + 1) {
			head = head.next;
			i++;
		}

		return head;
	}

	/**
	 * 
	 * 输入一个链表，反转链表后，输出链表的所有元素。
	 * 
	 * @param head
	 * @return
	 */
	public ListNode ReverseList(ListNode head) {

		ListNode tail = null, node = null;

		while (head != null) {
			node = new ListNode(head.val);
			node.next = tail;
			tail = node;
			head = head.next;
		}

		return tail;
	}

	/**
	 * 
	 * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
	 * 
	 * @param list1
	 * @param list2
	 * @return
	 */
	public ListNode Merge(ListNode list1, ListNode list2) {

		ListNode node = null, next;
		int val = 0;

		if (list1 != null && list2 != null) {

			int num = list1.val;
			if (list2.val < num) {
				num = list2.val;
				list2 = list2.next;
			} else
				list1 = list1.next;

			node = new ListNode(num);
		} else {
			return list1 == null ? list2 : list1;
		}

		next = node;
		while (list1 != null && list2 != null) {

			val = list1.val;

			if (list2.val < val) {
				val = list2.val;
				list2 = list2.next;
			} else
				list1 = list1.next;

			next.next = new ListNode(val);
			next = next.next;
		}

		if (list1 == null)
			next.next = list2;
		else
			next.next = list1;

		return node;
	}

	/**
	 * 
	 * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
	 * 
	 * @param root1
	 * @param root2
	 * @return
	 */
	public boolean HasSubtree(TreeNode root1, TreeNode root2) {
		boolean result = false;
		if (root1 != null && root2 != null) {
			if (root1.val == root2.val) {
				result = DoesTree1HaveTree2(root1, root2);
			}
			if (!result) {
				result = HasSubtree(root1.left, root2);
			}
			if (!result) {
				result = HasSubtree(root1.right, root2);
			}
		}
		return result;
	}

	public boolean DoesTree1HaveTree2(TreeNode root1, TreeNode root2) {
		if (root1 == null && root2 != null)
			return false;
		if (root2 == null)
			return true;
		if (root1.val != root2.val)
			return false;
		return DoesTree1HaveTree2(root1.left, root2.left) && DoesTree1HaveTree2(root1.right, root2.right);
	}

	/**
	 * 
	 * 操作给定的二叉树，将其变换为源二叉树的镜像。
	 * 
	 * @param root
	 */
	public void Mirror(TreeNode root) {

		TreeNode temp;
		if (root != null) {
			Mirror(root.left);
			Mirror(root.right);
			temp = root.left;
			root.left = root.right;
			root.right = temp;
		}
	}

	/**
	 * 
	 * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字
	 * 
	 * 例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字
	 * 1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
	 * 
	 * @param matrix
	 * @return
	 */
	public ArrayList<Integer> printMatrix(int[][] array) {
		ArrayList<Integer> result = new ArrayList<Integer>();

		if (array.length == 0)
			return result;

		int n = array.length, m = array[0].length;

		if (m == 0)
			return result;

		int layers = (Math.min(n, m) - 1) / 2 + 1;// 这个是层数

		for (int i = 0; i < layers; i++) {

			for (int k = i; k < m - i; k++)
				result.add(array[i][k]);// 左至右

			for (int j = i + 1; j < n - i; j++)
				result.add(array[j][m - i - 1]);// 右上至右下

			for (int k = m - i - 2; (k >= i) && (n - i - 1 != i); k--)
				result.add(array[n - i - 1][k]);// 右至左

			for (int j = n - i - 2; (j > i) && (m - i - 1 != i); j--)
				result.add(array[j][i]);// 左下至左上

		}
		return result;
	}

	/**
	 * 
	 * 定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
	 * 
	 * @param node
	 */
	/**
	 * Stack<Integer> stack = new Stack<Integer>(); public void push(int node) {
	 * stack.push(node); }
	 * 
	 * public void pop() { stack.pop(); }
	 * 
	 * public int top() { return stack.peek(); }
	 * 
	 * public int min() { int min = Integer.MAX_VALUE;
	 * 
	 * Stack<Integer> s = new Stack<Integer>();
	 * 
	 * while (!stack.isEmpty()){ int num = stack.pop(); if (num < min) min =
	 * num; s.push(num); }
	 * 
	 * while(!s.isEmpty()){ stack.push(s.pop()); }
	 * 
	 * return min; }
	 **/

	/**
	 * 
	 * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
	 * 假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,3,5,1,2是该压栈序列对应的一个弹出序列，
	 * 但4,3,5,1,2就不可能是该压栈序列的弹出序列。
	 * 
	 * （注意：这两个序列的长度是相等的）
	 * 
	 * @param pushA
	 * @param popA
	 * @return
	 */
	public boolean IsPopOrder(int[] pushA, int[] popA) {
		if (pushA.length == 0 || popA.length == 0) {
			return false;
		}
		Stack<Integer> stack = new Stack<Integer>();
		int j = 0;
		for (int i = 0; i < popA.length; i++) {
			stack.push(pushA[i]);
			while (j < popA.length && stack.peek() == popA[j]) {
				stack.pop();
				j++;
			}

		}
		return stack.empty() ? true : false;
	}

	/**
	 * 
	 * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
	 * 
	 * @param root
	 * @return
	 */
	public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {

		Queue<TreeNode> q = new LinkedList<TreeNode>();
		ArrayList<Integer> list = new ArrayList<Integer>();
		TreeNode node;

		if (root == null)
			return list;

		q.offer(root);

		while (!q.isEmpty()) {
			node = q.poll();
			if (node.left != null)
				q.offer(node.left);
			if (node.right != null)
				q.offer(node.right);
			list.add(node.val);
		}

		return list;
	}

	/**
	 * 
	 * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
	 * 
	 * @param sequence
	 * @return
	 */
	public boolean VerifySquenceOfBST(int[] sequence) {
		if (sequence.length == 0)
			return false;
		return IsTreeBST(sequence, 0, sequence.length - 1);
	}

	public boolean IsTreeBST(int[] sequence, int start, int end) {
		if (end <= start)
			return true;
		int i = start;
		for (; i < end; i++) {
			if (sequence[i] > sequence[end])
				break;
		}
		for (int j = i; j < end; j++) {
			if (sequence[j] < sequence[end])
				return false;
		}
		return IsTreeBST(sequence, start, i - 1) && IsTreeBST(sequence, i, end - 1);
	}

	/*************************************************************************************************/
	/**
	 * 输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
	 * 
	 * @param root
	 * @param target
	 * @return
	 */
	public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
		ArrayList<ArrayList<Integer>> arr = new ArrayList<ArrayList<Integer>>();
		if (root == null)
			return arr;
		ArrayList<Integer> a1 = new ArrayList<Integer>();
		int sum = 0;
		pa(root, target, arr, a1, sum);
		return arr;
	}

	public void pa(TreeNode root, int target, ArrayList<ArrayList<Integer>> arr, ArrayList<Integer> a1, int sum) {
		if (root == null)
			return;
		sum += root.val;

		if (root.left == null && root.right == null) {
			if (sum == target) {
				a1.add(root.val);
				arr.add(new ArrayList<Integer>(a1));
				a1.remove(a1.size() - 1);

			}
			return;

		}

		a1.add(root.val);
		pa(root.left, target, arr, a1, sum);
		pa(root.right, target, arr, a1, sum);
		a1.remove(a1.size() - 1);
	}

	/**
	 * 
	 * 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
	 * 返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
	 * 
	 * @param pHead
	 * @return
	 */
	public RandomListNode Clone(RandomListNode pHead) {
		RandomListNode p = pHead;
		RandomListNode t = pHead;
		while (p != null) {
			RandomListNode q = new RandomListNode(p.label);
			q.next = p.next;
			p.next = q;
			p = q.next;
		}
		while (t != null) {
			RandomListNode q = t.next;
			if (t.random != null)
				q.random = t.random.next;
			t = q.next;

		}
		RandomListNode s = new RandomListNode(0);
		RandomListNode s1 = s;
		while (pHead != null) {
			RandomListNode q = pHead.next;
			pHead.next = q.next;
			q.next = s.next;
			s.next = q;
			s = s.next;
			pHead = pHead.next;

		}
		return s1.next;

	}

	/**
	 * 
	 * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
	 * 
	 * @param root
	 * @return
	 */
	public TreeNode Convert(TreeNode root) {

		if (root == null)
			return null;
		if (root.left == null && root.right == null)
			return root;
		TreeNode left = Convert(root.left);
		TreeNode p = left;
		while (p != null && p.right != null) {
			p = p.right;
		}
		if (left != null) {
			p.right = root;
			root.left = p;
		}
		TreeNode right = Convert(root.right);
		if (right != null) {
			root.right = right;
			right.left = root;
		}

		return left != null ? left : root;
	}

	/**
	 * 
	 * 输入一个字符串,按字典序打印出该字符串中字符的所有排列。
	 * 
	 * 例如 输入 abc 输出 abc,acb,bac,bca,cab和cba
	 * 
	 * 注意:输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
	 * 
	 * @param str
	 * @return
	 */
	public ArrayList<String> Permutation(String str) {
		ArrayList<String> result = new ArrayList<String>();
		if (str == null || str.length() == 0) {
			return result;
		}

		char[] chars = str.toCharArray();
		TreeSet<String> temp = new TreeSet<>();
		Permutation(chars, 0, temp);
		result.addAll(temp);
		return result;
	}

	public void Permutation(char[] chars, int begin, TreeSet<String> result) {
		if (chars == null || chars.length == 0 || begin < 0 || begin > chars.length - 1) {
			return;
		}

		if (begin == chars.length - 1) {
			result.add(String.valueOf(chars));
		} else {
			for (int i = begin; i <= chars.length - 1; i++) {
				swap(chars, begin, i);

				Permutation(chars, begin + 1, result);

				swap(chars, begin, i);
			}
		}
	}

	public void swap(char[] x, int a, int b) {
		char t = x[a];
		x[a] = x[b];
		x[b] = t;
	}

	/**
	 * 
	 * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
	 * 由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
	 * 
	 * @param array
	 * @return
	 */
	public int MoreThanHalfNum_Solution(int[] array) {
		int len = array.length;
		if (len < 1) {
			return 0;
		}
		int count = 0;
		Arrays.sort(array);
		int num = array[len / 2];
		for (int i = 0; i < len; i++) {
			if (num == array[i])
				count++;
		}
		if (count <= (len / 2)) {
			num = 0;
		}
		return num;
	}

	/**
	 * 
	 * 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
	 * 
	 * @param input
	 * @param k
	 * @return
	 */
	public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
		ArrayList<Integer> list = new ArrayList<Integer>();

		if (input.length < k || k == 0)
			return list;

		for (int i = 0; i < k; i++)
			list.add(input[i]);

		for (int i = k; i < input.length; i++) {
			int j = this.getMax(list);
			int temp = (Integer) list.get(j);
			if (input[i] < temp)
				list.set(j, input[i]);
		}
		return list;
	}

	public int getMax(ArrayList<Integer> list) {
		int max = list.get(0);
		int j = 0;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) > max) {
				max = list.get(i);
				j = i;
			}
		}
		return j;
	}

	/**
	 * 
	 * HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。 今天测试组开完会后,他又发话了:在古老的一维模式识别中,
	 * 常常需要计算连续子向量的最大和,当向量全为正数的时候, 问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,
	 * 并期望旁边的正数会弥补它呢？
	 * 
	 * 例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。
	 * 你会不会被他忽悠住？(子向量的长度至少是1)
	 * 
	 * @param array
	 * @return
	 */
	public int FindGreatestSumOfSubArray(int[] array) {
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < array.length; i++) {
			int sum = 0;
			for (int j = i; j < array.length; j++) {
				sum += array[j];
				list.add(sum);
			}
		}
		if (list.size() <= 0)
			return 0;
		Collections.sort(list);
		return list.get(list.size() - 1);
	}

	/**
	 * 
	 * 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
	 * 为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,
	 * 但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化, 可以很快的求出任意非负整数区间中1出现的次数。
	 * 
	 * @param n
	 * @return
	 */
	public int NumberOf1Between1AndN_Solution(int n) {
		// 主要思路：设定整数点（如1、10、100等等）作为位置点i（对应n的各位、十位、百位等等），分别对每个数位上有多少包含1的点进行分析
		// 根据设定的整数位置，对n进行分割，分为两部分，高位n/i，低位n%i
		// 当i表示百位，且百位对应的数>=2,如n=31456,i=100，则a=314,b=56，此时百位为1的次数有a/10+1=32（最高两位0~31），每一次都包含100个连续的点，即共有(a%10+1)*100个点的百位为1
		// 当i表示百位，且百位对应的数为1，如n=31156,i=100，则a=311,b=56，此时百位对应的就是1，则共有a%10(最高两位0-30)次是包含100个连续点，当最高两位为31（即a=311），本次只对应局部点00~56，共b+1次，所有点加起来共有（a%10*100）+(b+1)，这些点百位对应为1
		// 当i表示百位，且百位对应的数为0,如n=31056,i=100，则a=310,b=56，此时百位为1的次数有a/10=31（最高两位0~30）
		// 综合以上三种情况，当百位对应0或>=2时，有(a+8)/10次包含所有100个点，还有当百位为1(a%10==1)，需要增加局部点b+1
		// 之所以补8，是因为当百位为0，则a/10==(a+8)/10，当百位>=2，补8会产生进位位，效果等同于(a/10+1)
		int count = 0;
		int i = 1;
		for (i = 1; i <= n; i *= 10) {
			// i表示当前分析的是哪一个数位
			int a = n / i, b = n % i;
			count = count + (a + 8) / 10 * i + (a % 10 == 1 ? (b + 1) : 0);
		}
		return count;
	}

	public int NumberOf1Between1AndN_Solution2(int n) {
		int count = 0;
		while (n > 0) {
			String str = String.valueOf(n);
			char[] chars = str.toCharArray();
			for (int i = 0; i < chars.length; i++) {
				if (chars[i] == '1')
					count++;
			}
			n--;
		}
		return count;
	}

	/**
	 * 
	 * 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
	 * 例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
	 * 
	 * @param numbers
	 * @return
	 */
	public String PrintMinNumber(int[] numbers) {
		int n;
		String s = "";
		ArrayList<Integer> list = new ArrayList<Integer>();
		n = numbers.length;

		for (int i = 0; i < n; i++) {
			list.add(numbers[i]);// 将数组放入arrayList中
		}
		// 实现了Comparator接口的compare方法，将集合元素按照compare方法的规则进行排序
		Collections.sort(list, new Comparator<Integer>() {

			@Override
			public int compare(Integer str1, Integer str2) {
				// TODO Auto-generated method stub
				String s1 = str1 + "" + str2;
				String s2 = str2 + "" + str1;

				return s1.compareTo(s2);
			}
		});

		for (int j : list) {
			s += j;
		}
		return s;
	}

	/**
	 * 
	 * 把只包含因子2、3和5的数称作丑数（Ugly Number）。 例如6、8都是丑数，但14不是，因为它包含因子7。
	 * 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数
	 * 
	 * @param index
	 * @return
	 */
	public int GetUglyNumber_Solution(int index) {

		if (index <= 0)
			return 0;
		int[] result = new int[index];
		int count = 0;
		int i2 = 0;
		int i3 = 0;
		int i5 = 0;

		result[0] = 1;
		int tmp = 0;
		while (count < index - 1) {
			tmp = min(result[i2] * 2, min(result[i3] * 3, result[i5] * 5));
			if (tmp == result[i2] * 2)
				i2++;// 三条if防止值是一样的，不要改成else的
			if (tmp == result[i3] * 3)
				i3++;
			if (tmp == result[i5] * 5)
				i5++;
			result[++count] = tmp;
		}
		return result[index - 1];
	}

	private int min(int a, int b) {
		return (a > b) ? b : a;
	}

	/**
	 * 
	 * 在一个字符串(1<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置
	 * 
	 * @param str
	 * @return
	 */
	public int FirstNotRepeatingChar(String str) {
		if (str == null || str.length() == 0)
			return -1;
		List<Character> list = new ArrayList<>();
		for (int i = 0; i < str.length(); i++) {
			char ch = str.charAt(i);
			if (!list.contains(ch)) {
				list.add(Character.valueOf(ch));
			} else {
				list.remove(Character.valueOf(ch));
			}
		}
		if (list.size() <= 0)
			return -1;
		return str.indexOf(list.get(0));
	}

	/**
	 * 
	 * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
	 * 输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
	 * 
	 * 
	 * 题目保证输入的数组中没有的相同的数字 数据范围： 对于%50的数据,size<=10^4 对于%75的数据,size<=10^5
	 * 对于%100的数据,size<=2*10^5
	 * 
	 * 例: 输入: 1,2,3,4,5,6,7,0 输出: 7
	 * 
	 * @param array
	 * @return
	 */
	public static int InversePairs(int[] array) {
		if (array == null || array.length == 0) {
			return 0;
		}
		int[] copy = new int[array.length];
		for (int i = 0; i < array.length; i++) {
			copy[i] = array[i];
		}
		int count = InversePairsCore(array, copy, 0, array.length - 1);// 数值过大求余
		return count;

	}

	private static int InversePairsCore(int[] array, int[] copy, int low, int high) {
		if (low == high) {
			return 0;
		}
		int mid = (low + high) >> 1;
		int leftCount = InversePairsCore(array, copy, low, mid) % 1000000007;
		int rightCount = InversePairsCore(array, copy, mid + 1, high) % 1000000007;
		int count = 0;
		int i = mid;
		int j = high;
		int locCopy = high;
		while (i >= low && j > mid) {
			if (array[i] > array[j]) {
				count += j - mid;
				copy[locCopy--] = array[i--];
				if (count >= 1000000007)// 数值过大求余
				{
					count %= 1000000007;
				}
			} else {
				copy[locCopy--] = array[j--];
			}
		}
		for (; i >= low; i--) {
			copy[locCopy--] = array[i];
		}
		for (; j > mid; j--) {
			copy[locCopy--] = array[j];
		}
		for (int s = low; s <= high; s++) {
			array[s] = copy[s];
		}
		return (leftCount + rightCount + count) % 1000000007;
	}

	/**
	 * 
	 * 输入两个链表，找出它们的第一个公共结点。
	 * 
	 * @param pHead1
	 * @param pHead2
	 * @return
	 */
	public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
		ListNode current1 = pHead1;
		ListNode current2 = pHead2;

		HashMap<ListNode, Integer> hashMap = new HashMap<ListNode, Integer>();
		while (current1 != null) {
			hashMap.put(current1, null);
			current1 = current1.next;
		}
		while (current2 != null) {
			if (hashMap.containsKey(current2))
				return current2;
			current2 = current2.next;
		}

		return null;

	}

	/**
	 * 
	 * 统计一个数字在排序数组中出现的次数
	 * 
	 * @param array
	 * @param k
	 * @return
	 */
	public int GetNumberOfK(int[] array, int k) {
		int count = 0;
		for (int i = 0; i < array.length && array[i] <= k; i++) {
			if (array[i] == k)
				count++;
		}
		return count;
	}

	/**
	 * 
	 * 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
	 * 
	 * @param root
	 * @return
	 */
	public int TreeDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}

		int nLelt = TreeDepth(root.left);
		int nRight = TreeDepth(root.right);

		return nLelt > nRight ? (nLelt + 1) : (nRight + 1);
	}

	/**
	 * 
	 * 输入一棵二叉树，判断该二叉树是否是平衡二叉树。
	 * 
	 * @param root
	 * @return
	 */
	public boolean IsBalanced_Solution(TreeNode root) {
		if (root == null)
			return true;

		if (Math.abs(getHeight(root.left) - getHeight(root.right)) > 1)
			return false;

		return IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);

	}

	public int getHeight(TreeNode root) {
		if (root == null)
			return 0;
		return max(getHeight(root.left), getHeight(root.right)) + 1;
	}

	private int max(int a, int b) {
		return (a > b) ? a : b;
	}

	/**
	 * 
	 * 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
	 * 
	 * num1,num2分别为长度为1的数组。传出参数 将num1[0],num2[0]设置为返回结果
	 * 
	 * @param array
	 * @param num1
	 * @param num2
	 */
	/*
	 * 考虑过程： 首先我们考虑这个问题的一个简单版本：一个数组里除了一个数字之外，其他的数字都出现了两次。请写程序找出这个只出现一次的数字。
	 * 这个题目的突破口在哪里？题目为什么要强调有一个数字出现一次，其他的出现两次？我们想到了异或运算的性质：任何一个数字异或它自己都等于0 。
	 * 也就是说，如果我们从头到尾依次异或数组中的每一个数字，那么最终的结果刚好是那个只出现一次的数字，因为那些出现两次的数字全部在异或中 抵消掉了。
	 * 
	 * 有了上面简单问题的解决方案之后，我们回到原始的问题。如果能够把原数组分为两个子数组。在每个子数组中，包含一个只出现一次的数字，
	 * 而其它数字都出现两次。如果能够这样拆分原数组，按照前面的办法就是分别求出这两个只出现一次的数字了。
	 * 
	 * 我们还是从头到尾依次异或数组中的每一个数字，那么最终得到的结果就是两个只出现一次的数字的异或结果。因为其它数字都出现了两次，
	 * 在异或中全部抵消掉了。由于这两个数字肯定不一样，那么这个异或结果肯定不为0 ，也就是说在这个结果数字的二进制表示中至少就有一位为1 。
	 * 我们在结果数字中找到第一个为1的位的位置，记为第N 位。现在我们以第N 位是不是1 为标准把原数组中的数字分成两个子数组，第一个子数组
	 * 中每个数字的第N 位都为1，而第二个子数组的每个数字的第N 位都为0 。
	 * 
	 * 现在我们已经把原数组分成了两个子数组，每个子数组都包含一个只出现一次的数字，而其它数字都出现了两次。 因此到此为止，所有的问题我们都已经解决。
	 */
	public static void findNumsAppearOnce(int[] array, int num1[], int num2[]) {
		if (array == null || array.length <= 1) {
			num1[0] = num2[0] = 0;
			return;
		}
		int len = array.length, index = 0, sum = 0;
		for (int i = 0; i < len; i++) {
			sum ^= array[i];
		}
		for (index = 0; index < 32; index++) {
			if ((sum & (1 << index)) != 0)
				break;
		}
		for (int i = 0; i < len; i++) {
			if ((array[i] & (1 << index)) != 0) {
				num2[0] ^= array[i];
			} else {
				num1[0] ^= array[i];
			}
		}
	}

	/**
	 * 数组a中只有一个数出现一次，其他数都出现了2次，找出这个数字
	 * 
	 * @param a
	 * @return
	 */
	public static int find1From2(int[] a) {
		int len = a.length, res = 0;
		for (int i = 0; i < len; i++) {
			res = res ^ a[i];
		}
		return res;
	}

	/**
	 * 数组a中只有一个数出现一次，其他数字都出现了3次，找出这个数字
	 * 
	 * @param a
	 * @return
	 */
	public static int find1From3(int[] a) {
		int[] bits = new int[32];
		int len = a.length;
		for (int i = 0; i < len; i++) {
			for (int j = 0; j < 32; j++) {
				bits[j] = bits[j] + ((a[i] >>> j) & 1);
			}
		}
		int res = 0;
		for (int i = 0; i < 32; i++) {
			if (bits[i] % 3 != 0) {
				res = res | (1 << i);
			}
		}
		return res;
	}

	/**
	 * 
	 * 小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。
	 * 但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。
	 * 没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。 现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列?
	 * Good Luck!
	 * 
	 * 输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
	 * 
	 * @param sum
	 * @return
	 */
	public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
		ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
		if (sum == 1)
			return list;
		int n = (int) (Math.ceil((Math.sqrt(8 * sum + 1) - 1) / 2));
		int tmp = 0;
		int num = 0;
		for (int i = n; i > 1; i--) {
			tmp = (i + 1) * i / 2;
			if ((sum - tmp) % i == 0) {
				ArrayList<Integer> arrayList = new ArrayList<Integer>();
				num = (sum - tmp) / i;
				for (int a = 1; a <= i; a++) {
					arrayList.add(num + a);
				}
				list.add(arrayList);
			}
		}
		return list;
	}

	// 根据数学公式计算:(a1+an)*n/2=s n=an-a1+1

	// (an+a1)*(an-a1+1)=2*s=k*l(k>l)

	// an=(k+l-1)/2 a1=(k-l+1)/2
	public ArrayList<ArrayList<Integer>> FindContinuousSequence2(int sum) {
		ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
		if (sum < 3)
			return list;
		int s = (int) Math.sqrt(2 * sum);
		for (int i = s; i >= 2; i--) {
			if (2 * sum % i == 0) {
				int d = 2 * sum / i;
				if (d % 2 == 0 && i % 2 != 0 || d % 2 != 0 && i % 2 == 0) {
					int a1 = (d - i + 1) / 2;
					int an = (d + i - 1) / 2;
					ArrayList<Integer> temp = new ArrayList<Integer>();
					for (int j = a1; j <= an; j++)
						temp.add(j);
					list.add(temp);
				}
			}
		}
		return list;
	}

	/**
	 * 
	 * 输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
	 * 
	 * @param array
	 * @param sum
	 * @return
	 */
	// a+b=sum,a和b越远乘积越小，而一头一尾两个指针往内靠近的方法找到的就是乘积最小的情况。
	public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		if (array == null || array.length < 2) {
			return list;
		}
		int i = 0, j = array.length - 1;
		while (i < j) {
			if (array[i] + array[j] == sum) {
				list.add(array[i]);
				list.add(array[j]);
				return list;
			} else if (array[i] + array[j] > sum) {
				j--;
			} else {
				i++;
			}

		}
		return list;
	}

	/**
	 * 
	 * 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。
	 * 对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,
	 * 要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
	 * 
	 * @param str
	 * @param n
	 * @return
	 */
	public String LeftRotateString(String str, int n) {
		if (str.length() == 0) {
			return str;
		}
		StringBuffer buffer = new StringBuffer(str);
		StringBuffer buffer1 = new StringBuffer(str);
		StringBuffer buffer2 = new StringBuffer();
		buffer.delete(0, n);
		buffer1.delete(n, str.length());
		buffer2.append(buffer.toString()).append(buffer1.toString());
		return buffer2.toString();
	}

	/**
	 * 
	 * 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。
	 * 同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。 例如，“student. a am
	 * I”。后来才意识到，这家伙原来把句子单词的顺序翻转了， 正确的句子应该是“I am a
	 * student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
	 * 
	 * @param str
	 * @return
	 */
	public String ReverseSentence(String str) {
		if (str.trim().equals("")) {
			return str;
		}
		String[] a = str.split(" ");
		StringBuffer o = new StringBuffer();
		int i;
		for (i = a.length; i > 0; i--) {
			o.append(a[i - 1]);
			if (i > 1) {
				o.append(" ");
			}
		}
		return o.toString();
	}

	/**
	 * 
	 * LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...
	 * 他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！
	 * “红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子..... LL不高兴了,他想了想,决定大\小
	 * 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。 上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So
	 * Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何。为了方便起见,你可以认为大小王是0。
	 * 
	 * @param numbers
	 * @return
	 */
	public boolean isContinuous(int[] numbers) {
		if (numbers == null)
			return false;
		Arrays.sort(numbers); // 先排序
		int zero = 0, i = 0;
		for (; i < numbers.length && numbers[i] == 0; i++) {
			zero++; // 统计0的个数
		}
		for (; i < numbers.length - 1 && zero >= 0; i++) {
			if (numbers[i] == numbers[i + 1]) // 有对子，则返回false
				return false;
			if (numbers[i] + 1 + zero >= numbers[i + 1]) { // 可以继续匹配
				zero -= numbers[i + 1] - numbers[i] - 1;
			} else
				return false;
		}
		if (i == numbers.length - 1)
			return true;
		else
			return false;
	}

	/**
	 * 
	 * 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。 HF作为牛客的资深元老,自然也准备了一些小游戏。
	 * 其中,有个游戏是这样的: 首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。
	 * 每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
	 * 从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,
	 * 并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。 请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
	 * 
	 * @param n
	 * @param m
	 * @return
	 */
	public int LastRemaining_Solution(int n, int m) {
		if (n == 0 || m == 0)
			return -1;
		int s = 0;
		for (int i = 2; i <= n; i++) {
			s = (s + m) % i;
		}
		return s;
	}

	/*
	 * 如果只求最后一个报数胜利者的话，我们可以用数学归纳法解决该问题，为了讨 论方便，先把问题稍微改变一下，并不影响原意：
	 * 问题描述：n个人（编号0~(n-1))，从0开始报数，报到(m-1)的退出，剩下的人 继续从0开始报数。求胜利者的编号。
	 * 我们知道第一个人(编号一定是m%n-1) 出列之后，剩下的n-1个人组成了一个新 的约瑟夫环（以编号为k=m%n的人开始）: k k+1 k+2
	 * ... n-2, n-1, 0, 1, 2, ... k-2并且从k开始报0。 现在我们把他们的编号做一下转换： k --> 0 k+1 -->
	 * 1 k+2 --> 2 ... ... k-2 --> n-2 k-1 --> n-1
	 * 变换后就完完全全成为了(n-1)个人报数的子问题，假如我们知道这个子问题的解：
	 * 例如x是最终的胜利者，那么根据上面这个表把这个x变回去不刚好就是n个人情
	 * 况的解吗？！！变回去的公式很简单，相信大家都可以推出来：x'=(x+k)%n。
	 * 令f[i]表示i个人玩游戏报m退出最后胜利者的编号，最后的结果自然是f[n]。 递推公式 f[1]=0; f[i]=(f[i-1]+m)%i;
	 * (i>1) 有了这个公式，我们要做的就是从1-n顺序算出f[i]的数值，最后结果是f[n]。
	 * 因为实际生活中编号总是从1开始，我们输出f[n]+1。
	 */
	public int LastRemaining_Solution2(int n, int m) {
		if (n == 0)
			return -1;
		if (n == 1)
			return 0;
		else
			return (LastRemaining_Solution(n - 1, m) + m) % n;
	}

	/**
	 * 
	 * 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
	 * 
	 * @param n
	 * @return
	 */
	public int Sum_Solution(int n) {
		int sum = n;
		boolean ans = (n > 0) && ((sum += Sum_Solution(n - 1)) > 0);
		return sum;
	}

	/**
	 * 
	 * 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
	 * 
	 * @param num1
	 * @param num2
	 * @return
	 */
	public int Add(int num1, int num2) {
		while (num2 != 0) {
			int temp = num1 ^ num2;
			num2 = (num1 & num2) << 1;
			num1 = temp;
		}
		return num1;
	}

	/**
	 * 
	 * 将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
	 * 
	 * 输入描述: 输入一个字符串,包括数字字母符号,可以为空 输出描述: 如果是合法的数值表达则返回该数字，否则返回0
	 * 
	 * 输入 +2147483647 1a33 输出
	 * 
	 * 2147483647 0
	 * 
	 * @param str
	 * @return
	 */
	public int StrToInt(String str) {
		if (str.equals("") || str.length() == 0)
			return 0;
		char[] a = str.toCharArray();
		int fuhao = 0;
		if (a[0] == '-')
			fuhao = 1;
		int sum = 0;
		for (int i = fuhao; i < a.length; i++) {
			if (a[i] == '+')
				continue;
			if (a[i] < 48 || a[i] > 57)
				return 0;
			sum = sum * 10 + a[i] - 48;
		}
		return fuhao == 0 ? sum : sum * -1;
	}

	/**
	 * 
	 * 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。
	 * 也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，
	 * 那么对应的输出是第一个重复的数字2。
	 * 
	 * @param numbers
	 *            an array of integers
	 * @param length
	 *            the length of array numbers
	 * @param duplication
	 *            (Output) the duplicated number in the array number, length of
	 *            duplication array is 1,so using duplication[0] = ? in
	 *            implementation; 这里要特别注意:返回任意重复的一个，赋值duplication[0]
	 * @return true if the input is valid, and there are some duplications in
	 *         the array number otherwise false
	 */
	public boolean duplicate(int numbers[], int length, int[] duplication) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < length; i++) {
			sb.append(numbers[i] + "");
		}
		for (int j = 0; j < length; j++) {
			if (sb.indexOf(numbers[j] + "") != sb.lastIndexOf(numbers[j] + "")) {
				duplication[0] = numbers[j];
				return true;
			}
		}
		return false;
	}

	/**
	 * 
	 * 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
	 * 其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
	 * 
	 * @param A
	 * @return
	 */
	public int[] multiply(int[] A) {
		if (A.length <= 0) {
			return A;
		}
		int sum = 1;
		for (int i = 0; i < A.length; i++) {
			sum *= A[i];
		}
		if (sum == 0) {
			sum = 1;
			int flag = 0;
			int num = 0;
			for (int i = 0; i < A.length; i++) {
				if (!(A[i] == 0) && flag < 2) {
					sum *= A[i];
				} else if (A[i] == 0 && flag < 2) {
					num = i;
					flag++;
				} else {
					break;
				}
			}

			for (int i = 0; i < A.length; i++) {
				A[i] = 0;
			}
			if (flag < 2) {
				A[num] = sum;
			}
			return A;
		}
		int[] B = A;
		for (int i = 0; i < A.length; i++) {
			B[i] = (int) (sum * Math.pow(A[i], -1));
		}
		return B;
	}

	/**
	 * 
	 * 请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符， 而'*'表示它前面的字符可以出现任意次（包含0次）。
	 * 在本题中，匹配是指字符串的所有字符匹配整个模式。
	 * 例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
	 * 
	 * @param str
	 * @param pattern
	 * @return
	 */
	public boolean match(char[] str, char[] pattern) {
		if (str == null || pattern == null) {
			return false;
		}
		int strIndex = 0;
		int patternIndex = 0;
		return matchCore(str, strIndex, pattern, patternIndex);
	}

	public boolean matchCore(char[] str, int strIndex, char[] pattern, int patternIndex) {
		// 有效性检验：str到尾，pattern到尾，匹配成功
		if (strIndex == str.length && patternIndex == pattern.length) {
			return true;
		}
		// pattern先到尾，匹配失败
		if (strIndex != str.length && patternIndex == pattern.length) {
			return false;
		}
		// 模式第2个是*，且字符串第1个跟模式第1个匹配,分3种匹配模式；如不匹配，模式后移2位
		if (patternIndex + 1 < pattern.length && pattern[patternIndex + 1] == '*') {
			if ((strIndex != str.length && pattern[patternIndex] == str[strIndex])
					|| (pattern[patternIndex] == '.' && strIndex != str.length)) {
				return matchCore(str, strIndex, pattern, patternIndex + 2)// 模式后移2，视为x*匹配0个字符
						|| matchCore(str, strIndex + 1, pattern, patternIndex + 2)// 视为模式匹配1个字符
						|| matchCore(str, strIndex + 1, pattern, patternIndex);// *匹配1个，再匹配str中的下一个
			} else {
				return matchCore(str, strIndex, pattern, patternIndex + 2);
			}
		}
		// 模式第2个不是*，且字符串第1个跟模式第1个匹配，则都后移1位，否则直接返回false
		if ((strIndex != str.length && pattern[patternIndex] == str[strIndex])
				|| (pattern[patternIndex] == '.' && strIndex != str.length)) {
			return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
		}
		return false;
	}

	/**
	 * 
	 * 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
	 * 例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。
	 * 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
	 * 
	 * @param str
	 * @return
	 */
	public boolean isNumeric(char[] str) {
		String s = String.valueOf(str);

		return s.matches("[+-]?[0-9]*(\\.[0-9]*)?([eE][+-]?[0-9]+)?");
	}

	/**
	 * 
	 * 请实现一个函数用来找出字符流中第一个只出现一次的字符。 例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
	 * 当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
	 * 
	 * @param ch
	 */
	int count[] = new int[256];
	// Insert one char from stringstream
	int index = 1;

	public void Insert(char ch) {
		if (count[ch] == 0) {
			count[ch] = index++;
		} else {
			count[ch] = -1;
		}
	}

	// return the first appearence once char in current stringstream
	public char FirstAppearingOnce() {
		int temp = Integer.MAX_VALUE;
		char ch = '#';
		for (int i = 0; i < 256; i++) {
			if (count[i] != 0 && count[i] != -1 && count[i] < temp) {
				temp = count[i];
				ch = (char) i;
			}
		}
		return ch;
	}

	/**
	 * 
	 * 一个链表中包含环，请找出该链表的环的入口结点。
	 * 
	 * @param pHead
	 * @return
	 */
	public ListNode EntryNodeOfLoop(ListNode pHead) {
		if (pHead == null || pHead.next == null)
			return null;
		ListNode p1 = pHead;
		ListNode p2 = pHead;
		while (p2 != null && p2.next != null) {
			p1 = p1.next;
			p2 = p2.next.next;
			if (p1 == p2) {
				p1 = pHead;
				while (p1 != p2) {
					p1 = p1.next;
					p2 = p2.next;
				}
				if (p1 == p2)
					return p1;
			}
		}
		return null;
	}

	/**
	 * 
	 * 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5
	 * 处理后为 1->2->5
	 * 
	 * @param pHead
	 * @return
	 */
	public ListNode deleteDuplication(ListNode pHead) {

		if (pHead == null)
			return null;
		ListNode p = pHead;
		ListNode n = new ListNode(0);
		ListNode pre = n;
		n.next = pHead;
		boolean flag = false;
		while (p != null) {
			ListNode q = p.next;
			if (q == null)
				break;
			if (q.val == p.val) {
				while (q != null && q.val == p.val) {
					q = q.next;
				}
				pre.next = q;
				p = q;
			} else {
				if (!flag) {
					n.next = p;
					flag = true;
				}
				pre = p;
				p = q;
			}
		}
		return n.next;
	}

	/**
	 * 
	 * 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。 注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
	 * 
	 * @param pNode
	 * @return
	 */
	public TreeLinkNode GetNext(TreeLinkNode node) {
		if (node == null)
			return null;
		if (node.right != null) {
			node = node.right;
			while (node.left != null) {
				node = node.left;

			}
			return node;
		}
		while (node.next != null) {
			if (node.next.left == node)
				return node.next;
			node = node.next;
		}
		return null;
	}

	/**
	 * 
	 * 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
	 * 
	 * @param pRoot
	 * @return
	 */
	boolean isSymmetrical(TreeNode pRoot) {
		TreeNode node = getMirror(pRoot);
		return isSymmetrical(pRoot, node);
	}

	boolean isSymmetrical(TreeNode pRoot, TreeNode node) {
		if (pRoot == null && node == null) {
			return true;
		} else if (pRoot == null || node == null) {
			return false;
		}
		if (pRoot.val == node.val) {
			return isSymmetrical(pRoot.left, node.left) && isSymmetrical(pRoot.right, node.right);
		}
		return false;
	}

	TreeNode getMirror(TreeNode pRoot) {
		if (pRoot == null) {
			return null;
		}
		TreeNode root = new TreeNode(pRoot.val);
		root.right = getMirror(pRoot.left);
		root.left = getMirror(pRoot.right);
		return root;
	}

	/**
	 * 
	 * 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印， 第三行按照从左到右的顺序打印，其他行以此类推。
	 * 
	 * @param pRoot
	 * @return
	 */
	public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
		ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
		if (pRoot == null) {
			return ret;
		}
		ArrayList<Integer> list = new ArrayList<>();
		LinkedList<TreeNode> queue = new LinkedList<>();
		queue.addLast(null);// 层分隔符
		queue.addLast(pRoot);
		boolean leftToRight = true;

		while (queue.size() != 1) {
			TreeNode node = queue.removeFirst();
			if (node == null) {// 到达层分隔符
				Iterator<TreeNode> iter = null;
				if (leftToRight) {
					iter = queue.iterator();// 从前往后遍历
				} else {
					iter = queue.descendingIterator();// 从后往前遍历
				}
				leftToRight = !leftToRight;
				while (iter.hasNext()) {
					TreeNode temp = (TreeNode) iter.next();
					list.add(temp.val);
				}
				ret.add(new ArrayList<Integer>(list));
				list.clear();
				queue.addLast(null);// 添加层分隔符
				continue;// 一定要continue
			}
			if (node.left != null) {
				queue.addLast(node.left);
			}
			if (node.right != null) {
				queue.addLast(node.right);
			}
		}

		return ret;
	}

	/**
	 * 
	 * 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
	 * 
	 * @param pRoot
	 * @return
	 */
	ArrayList<ArrayList<Integer>> Print2(TreeNode pRoot) {
		ArrayList<ArrayList<Integer>> ret = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		LinkedList<TreeNode> q = new LinkedList<TreeNode>();
		if (pRoot == null)
			return ret;
		q.add(pRoot);
		int now = 1, next = 0;
		while (!q.isEmpty()) {
			TreeNode t = q.remove();
			now--;
			tmp.add(t.val);
			if (t.left != null) {
				q.add(t.left);
				next++;
			}
			if (t.right != null) {
				q.add(t.right);
				next++;
			}
			if (now == 0) {
				ret.add(new ArrayList<Integer>(tmp));
				tmp.clear();
				now = next;
				next = 0;
			}
		}
		return ret;
	}

	/**
	 * 
	 * 请实现两个函数，分别用来序列化和反序列化二叉树
	 * 
	 * @param root
	 * @return
	 */
	String Serialize(TreeNode root) {
		if (root == null)
			return "";
		StringBuilder sb = new StringBuilder();
		Serialize2(root, sb);
		return sb.toString();
	}

	void Serialize2(TreeNode root, StringBuilder sb) {
		if (root == null) {
			sb.append("#,");
			return;
		}
		sb.append(root.val);
		sb.append(',');
		Serialize2(root.left, sb);
		Serialize2(root.right, sb);
	}

	int index2 = -1;

	TreeNode Deserialize(String str) {
		if (str.length() == 0)
			return null;
		String[] strs = str.split(",");
		return Deserialize2(strs);
	}

	TreeNode Deserialize2(String[] strs) {
		index2++;
		if (!strs[index2].equals("#")) {
			TreeNode root = new TreeNode(0);
			root.val = Integer.parseInt(strs[index2]);
			root.left = Deserialize2(strs);
			root.right = Deserialize2(strs);
			return root;
		}
		return null;
	}

	/**
	 * 
	 * 给定一颗二叉搜索树，请找出其中的第k大的结点。例如， 5 / \ 3 7 /\ /\ 2 4 6 8 中，按结点数值大小顺序第三个结点的值为4。
	 * 
	 * @param pRoot
	 * @param k
	 * @return
	 */
	TreeNode KthNode(TreeNode pRoot, int k) {
		if (pRoot == null || k <= 0) {
			return null;
		}
		TreeNode[] result = new TreeNode[1];
		KthNode(pRoot, k, new int[1], result);
		return result[0];
	}

	void KthNode(TreeNode pRoot, int k, int[] count, TreeNode[] result) {
		if (result[0] != null || pRoot == null) {
			return;
		}
		KthNode(pRoot.left, k, count, result);
		count[0]++;
		if (count[0] == k) {
			result[0] = pRoot;
		}
		KthNode(pRoot.right, k, count, result);
	}

	/**
	 * 
	 * 如何得到一个数据流中的中位数？ 如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
	 * 如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
	 * 
	 * @param num
	 */
	LinkedList<Integer> list = new LinkedList<Integer>();

	public void Insert(Integer num) {
		if (list.size() == 0 || num < list.getFirst()) {
			list.addFirst(num);
		} else {
			boolean insertFlag = false;
			for (Integer e : list) {
				if (num < e) {
					int index = list.indexOf(e);
					list.add(index, num);
					insertFlag = true;
					break;
				}
			}
			if (!insertFlag) {
				list.addLast(num);
			}
		}

	}

	public Double GetMedian() {
		if (list.size() == 0) {
			return null;
		}

		if (list.size() % 2 == 0) {
			int i = list.size() / 2;
			Double a = Double.valueOf(list.get(i - 1) + list.get(i));
			return a / 2;
		}
		list.get(0);
		Double b = Double.valueOf(list.get((list.size() + 1) / 2 - 1));
		return Double.valueOf(list.get((list.size() + 1) / 2 - 1));

	}

	/**
	 * 
	 * 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
	 * 
	 * 例如， 如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，
	 * 那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个：
	 * {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}，
	 * {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
	 * 
	 * @param num
	 * @param size
	 * @return
	 */
	public ArrayList<Integer> maxInWindows(int[] num, int size) {
		if (num == null || size < 0) {
			return null;
		}
		ArrayList<Integer> list = new ArrayList<Integer>();
		if (size == 0) {
			return list;
		}
		ArrayList<Integer> temp = null;
		int length = num.length;
		if (length < size) {
			return list;
		} else {
			for (int i = 0; i < length - size + 1; i++) {
				temp = new ArrayList<Integer>();
				for (int j = i; j < size + i; j++) {
					temp.add(num[j]);
				}
				Collections.sort(temp);
				list.add(temp.get(temp.size() - 1));
			}
		}
		return list;
	}

	/**
	 * 
	 * 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
	 * 路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
	 * 如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。
	 * 
	 * 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，
	 * 但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行 第二个格子之后，路径不能再次进入该格子。
	 * 
	 * @param matrix
	 * @param rows
	 * @param cols
	 * @param str
	 * @return
	 */
	public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
		if (matrix == null || matrix.length == 0 || str == null || str.length == 0 || matrix.length != rows * cols
				|| rows <= 0 || cols <= 0 || rows * cols < str.length) {
			return false;
		}

		boolean[] visited = new boolean[rows * cols];
		int[] pathLength = { 0 };

		for (int i = 0; i <= rows - 1; i++) {
			for (int j = 0; j <= cols - 1; j++) {
				if (hasPathCore(matrix, rows, cols, str, i, j, visited, pathLength)) {
					return true;
				}
			}
		}

		return false;
	}

	public boolean hasPathCore(char[] matrix, int rows, int cols, char[] str, int row, int col, boolean[] visited,
			int[] pathLength) {
		boolean flag = false;

		if (row >= 0 && row < rows && col >= 0 && col < cols && !visited[row * cols + col]
				&& matrix[row * cols + col] == str[pathLength[0]]) {
			pathLength[0]++;
			visited[row * cols + col] = true;
			if (pathLength[0] == str.length) {
				return true;
			}
			flag = hasPathCore(matrix, rows, cols, str, row, col + 1, visited, pathLength)
					|| hasPathCore(matrix, rows, cols, str, row + 1, col, visited, pathLength)
					|| hasPathCore(matrix, rows, cols, str, row, col - 1, visited, pathLength)
					|| hasPathCore(matrix, rows, cols, str, row - 1, col, visited, pathLength);

			if (!flag) {
				pathLength[0]--;
				visited[row * cols + col] = false;
			}
		}

		return flag;
	}

	/**
	 * 
	 * 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动， 每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标
	 * 和列坐标的数位之和大于k的格子。
	 * 
	 * 例如，当k为18时，机器人能够进入方格（35,37）， 因为3+5+3+7 = 18。但是，它不能进入方格（35,38）， 因为3+5+3+8 =
	 * 19。请问该机器人能够达到多少个格子？
	 * 
	 * @param threshold
	 * @param rows
	 * @param cols
	 * @return
	 */
	public int movingCount(int threshold, int rows, int cols) {
		boolean[] visited = new boolean[rows * cols];
		return movingCountCore(threshold, rows, cols, 0, 0, visited);
	}

	private int movingCountCore(int threshold, int rows, int cols, int row, int col, boolean[] visited) {
		if (row < 0 || row >= rows || col < 0 || col >= cols)
			return 0;
		int i = row * cols + col;
		if (visited[i] || !checkSum(threshold, row, col))
			return 0;
		visited[i] = true;
		return 1 + movingCountCore(threshold, rows, cols, row, col + 1, visited)
				+ movingCountCore(threshold, rows, cols, row, col - 1, visited)
				+ movingCountCore(threshold, rows, cols, row + 1, col, visited)
				+ movingCountCore(threshold, rows, cols, row - 1, col, visited);
	}

	private boolean checkSum(int threshold, int row, int col) {
		int sum = 0;
		while (row != 0) {
			sum += row % 10;
			row = row / 10;
		}
		while (col != 0) {
			sum += col % 10;
			col = col / 10;
		}
		if (sum > threshold)
			return false;
		return true;
	}
}