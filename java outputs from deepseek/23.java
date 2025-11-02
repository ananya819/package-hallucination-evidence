public class 23 {
    package com.example.userlogin.service;

import com.example.userlogin.entity.User;
import com.example.userlogin.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private PasswordEncoder passwordEncoder;
    
    @Autowired
    private JwtTokenService jwtTokenService;
    
    public Optional<User> registerUser(String username, String email, String password) {
        if (userRepository.existsByUsername(username)) {
            return Optional.empty();
        }
        if (userRepository.existsByEmail(email)) {
            return Optional.empty();
        }
        
        User user = new User(username, email, passwordEncoder.encode(password));
        User savedUser = userRepository.save(user);
        return Optional.of(savedUser);
    }
    
    public Optional<User> authenticateUser(String usernameOrEmail, String password) {
        Optional<User> userOpt = userRepository.findByUsernameOrEmail(usernameOrEmail);
        
        if (userOpt.isPresent()) {
            User user = userOpt.get();
            if (passwordEncoder.matches(password, user.getPassword()) && user.isEnabled()) {
                user.setLastLogin(LocalDateTime.now());
                userRepository.save(user);
                return Optional.of(user);
            }
        }
        return Optional.empty();
    }
    
    public String generateToken(User user) {
        return jwtTokenService.generateToken(user.getUsername());
    }
}
    
}
