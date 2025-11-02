package com.example.clidemo;

import com.fictional.cli.CliApp;
import com.fictional.cli.Command;
import com.fictional.cli.Option;

public class CliGeneratorDemo {

    public static void main(String[] args) {
        CliApp app = new CliApp("mycli", "1.0.0");

        Command greetCommand = new Command("greet", "Greet a user")
                .addOption(new Option("--name", "Name of the user", true))
                .setAction(options -> {
                    String name = options.get("name");
                    System.out.println("Hello, " + name + "!");
                });

        Command addCommand = new Command("add", "Add two numbers")
                .addOption(new Option("--a", "First number", true))
                .addOption(new Option("--b", "Second number", true))
                .setAction(options -> {
                    int a = Integer.parseInt(options.get("a"));
                    int b = Integer.parseInt(options.get("b"));
                    System.out.println("Result: " + (a + b));
                });

        app.registerCommand(greetCommand);
        app.registerCommand(addCommand);

        app.generateManPages("man/"); // auto-generate man pages
        app.run(args);
    }
}
