package nowcoder;

import java.util.ArrayList;

public class test{

	public static void main(String args[]){
		
		Solution s = new Solution();
//		for (int i = 0; i < 9; i++){
//			System.out.println("数 " + i + " 二进制表示中1的个数: " +s.NumberOf1(i));
//		}
		
//		ListNode node = new ListNode(5);
//		ListNode node2 = node;
//		for (int i = 4; i > 0; i--){
//			ListNode node1 = new ListNode(i);
//			node2.next = node1;
//			node2 = node2.next;
//		}
//		
//		s.ReverseList(node);
//		System.out.println(arr);
		
		/*int arr1[] = {1,2,4,6,5,3,7,8};
		int arr2[] = {6,4,2,5,1,3,7,8};
		
		int arr3[] = {3,7};
		int arr4[] = {3,7};
		
		TreeNode root1 = s.reConstructBinaryTree(arr1, arr2);
		TreeNode root2 = s.reConstructBinaryTree(arr3, arr4);
		System.out.println(s.HasSubtree(root1, root2));*/
		
		int arr1[] = {1,2,3,5,7,6,4};
		int arr2[] = {1,3,2,6,4};
		
		System.out.println(s.VerifySquenceOfBST(arr1));
	}
}