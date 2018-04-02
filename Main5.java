package nowcoder;

import java.util.Scanner;

public class Main5{
	public static void main(String args[]){
		Scanner sc = new Scanner(System.in);
		
		int t = sc.nextInt();
		
		for (int i = 0; i < t; i++){
			int n = sc.nextInt();
			
			int num1 = 0;
			int num2 = 0;
			int num3 = 0;
			
			for (int j = 0; j < n; j++){
				int num =  sc.nextInt();
				
				if (num % 4 != 0){
					num1++;
				}else if (num % 4 != 0  && num / 2 == 0){
					num2++;
				}else{
					num3++;
				}
			}
			
			int size1 = num1 + num2 / 2;
			int size2 = num2;
			
			System.out.println(size1 == size2 ||  size1 == size2 + 1 ? "Yes" : "No");
		}
	}
}