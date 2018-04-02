package nowcoder;

import java.util.Scanner;
import java.util.Stack;

public class Main {
	public static void main(String args[]) {

		Scanner sc = new Scanner(System.in);
		String s = "";

		int n = sc.nextInt();

		while (n != 0) {
			if (n % 2 == 0) {
				s = "2" + s;
				n  = (n - 2) / 2;
			} else {
				s = "1" + s;
				n = (n - 1) / 2;
			}
		}
		
		System.out.print(s);
	}
}