/**
 * Template Router for HTML views
 */
@URL("/web")
public class TemplateRouter implements App {
    
    @GET
    @URL("/users")
    public IResponse userListView(IRequest request, ITemplateEngine templateEngine) {
        try {
            List<Django4jRESTApp.User> users = new ArrayList<>(Django4jRESTApp.userDatabase.values());
            
            Template template = templateEngine.getTemplate("users.html");
            Map<String, Object> context = new HashMap<>();
            context.put("users", users);
            context.put("title", "User Management");
            context.put("currentTime", new Date());
            
            return IResponse.html(200, template.render(context));
            
        } catch (Exception e) {
            return IResponse.html(500, "<h1>Error loading template</h1><p>" + e.getMessage() + "</p>");
        }
    }
    
    @GET
    @URL("/users/{id}")
    public IResponse userDetailView(IRequest request, @PathParam("id") int userId, ITemplateEngine templateEngine) {
        try {
            Django4jRESTApp.User user = Django4jRESTApp.userDatabase.get(userId);
            if (user == null) {
                return IResponse.html(404, "<h1>User not found</h1>");
            }
            
            Template template = templateEngine.getTemplate("user_detail.html");
            Map<String, Object> context = new HashMap<>();
            context.put("user", user);
            context.put("title", "User: " + user.getUsername());
            
            return IResponse.html(200, template.render(context));
            
        } catch (Exception e) {
            return IResponse.html(500, "<h1>Error loading template</h1><p>" + e.getMessage() + "</p>");
        }
    }
}