package nowcoder;

import java.util.Scanner;

public class Main2{
	public static void main(String args[]){
		Scanner sc = new Scanner(System.in);
		
		int num = sc.nextInt();
		
		int num2 = 0;
		
		int n = num;
		while(n / 10 != 0){
			num2 = num2 * 10 + n % 10;
			n = n / 10;
		}
		
		System.out.println(num + num2 * 10 + n);
	}
}