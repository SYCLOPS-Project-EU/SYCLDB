plugins {
    id("java")
    id("idea")
    application
}

group = "com.eurecom.calcite"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    implementation("org.apache.calcite:calcite-core:1.39.0")

    // https://mvnrepository.com/artifact/org.apache.thrift/libthrift
    implementation("org.apache.thrift:libthrift:0.21.0")

    // https://mvnrepository.com/artifact/javax.annotation/javax.annotation-api
    implementation("javax.annotation:javax.annotation-api:1.3.2")

    // https://mvnrepository.com/artifact/org.json/json
    implementation("org.json:json:20250107")
}

tasks.test {
    useJUnitPlatform()
}

idea {
    module {
        isDownloadJavadoc = true
        isDownloadSources = true
    }
}

application {
    mainClass = "com.eurecom.calcite.thrift.ServerCaller"
}