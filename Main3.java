package nowcoder;

import java.util.Scanner;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class Main3 {
	public static void main(String[] args) {

		Scanner sc = new Scanner(System.in);

		String line = sc.nextLine();
		
		ArrayList<String> list = new ArrayList<>();
		Pattern p = Pattern.compile("([a-z])\\1*");
		Matcher m = p.matcher(line);

		while(m.find()) {
			list.add(m.group());
		}

		float sumLen = 0;

		for(String str : list) {
			sumLen += str.length();
		}

		DecimalFormat df = new DecimalFormat("######0.00");   
		System.out.println(df.format(sumLen/list.size()));
	}
}