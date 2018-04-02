package nowcoder;

import java.util.HashMap;
import java.util.Scanner;

public class Main4{
	public static void main(String args[]){
		Scanner sc = new Scanner(System.in);
		
		int n = sc.nextInt();
		
		int[] v = new int[n];
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		
		for (int i = 0; i < n; i++){
			int num = sc.nextInt();
			if (map.containsKey(num)){
				map.replace(num, map.get(num) + 1);
			}else{
				map.put(num, 1);
			}
		}
		
		int num1 = map.get(1);
		int num2 = map.get(2);
		int num5 = map.get(5);
		int num6 = map.get(6);
		
		if (num1 /2 >= num2){
			num1  = num1 % 2 + num1 /2 - num2;
			num2 = 0;
		}else{
			num1 = num1 % 2 + num2 -num1 /2;
			num2 = num2 - num1 /2;
		}
		
		if (num5 >= num6){
			num5 -= num6;
			num6 = 0;
		}
	}
}