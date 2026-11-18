pm.test("Unsupported oneOf returns HTTP 400", () => pm.response.to.have.status(400));
pm.test("Public error names the schema field and constraint", () => {
    const error = pm.response.json().error;
    pm.expect(error.code).to.eql("invalid_schema");
    pm.expect(error.param).to.eql("response_format");
    pm.expect(error.message).to.include("oneOf");
});
