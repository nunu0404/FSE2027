# Complementarity Case Studies

## RF_correct__VLM_correct
### full_rq0_seed42__rq0_0009__rq0_0094
- snippets: `rq0_0009` vs `rq0_0094`
- difficulty: `medium`
- human z: 0.0974 vs 0.8348; gold: `rq0_0094`
- RF: score_a=0.4553, score_b=0.4560, margin=-0.0006, pred=`rq0_0094`, correct=True
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0094`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0009.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0094.png`

Code A excerpt:
```java
    private void moveUnit(KeyEvent e) {
        if (!parent.isMapboardActionsEnabled()) {
            return;
        }
        
        switch (e.getKeyCode()) {
        case KeyEvent.VK_ESCAPE:
            // main menu
            break;
        case KeyEvent.VK_NUMPAD1:
        case KeyEvent.VK_END:
            inGameController.moveActiveUnit(Map.SW);

```
Code B excerpt:
```java
    /**
     * Applies this action.
     * 
     * @param e The <code>ActionEvent</code>.
     */
    public void actionPerformed(ActionEvent e) {
        final Game game = freeColClient.getGame();
        final Map map = game.getMap();

        Parameters p = showParametersDialog();

```

## RF_correct__VLM_wrong
### full_rq0_seed42__rq0_0263__rq0_0303
- snippets: `rq0_0263` vs `rq0_0303`
- difficulty: `medium`
- human z: -1.0949 vs -0.1874; gold: `rq0_0303`
- RF: score_a=-0.6307, score_b=-0.6230, margin=-0.0077, pred=`rq0_0303`, correct=True
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0263`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0263.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0303.png`

Code A excerpt:
```java
@SuppressWarnings( {"SimplifiableIfStatement"})
	private boolean isUnequivocallyNonDirty(Object entity) {

		if(entity instanceof SelfDirtinessTracker)
			return ((SelfDirtinessTracker) entity).$$_hibernate_hasDirtyAttributes();

		final CustomEntityDirtinessStrategy customEntityDirtinessStrategy =
				persistenceContext.getSession().getFactory().getCustomEntityDirtinessStrategy();
		if ( customEntityDirtinessStrategy.canDirtyCheck( entity, getPersister(), (Session) persistenceContext.getSession() ) ) {
			return ! customEntityDirtinessStrategy.isDirty( entity, getPersister(), (Session) persistenceContext.getSession() );
		}

		if ( getPersister().hasMutableProperties() ) {
			return false;
		}

		if ( getPersister().getInstrumentationMetadata().isInstrumented() ) {
			// the entity must be instrumented (otherwise we cant check dirty flag) and the dirty flag is false
			return ! getPersister().getInstrumentationMetadata().extractInterceptor( entity ).isDirty();
		}

		return false;
	}
```
Code B excerpt:
```java
@Override
	public void startingCollectionElements(CollectionElementDefinition elementDefinition) {
		final Type elementType = elementDefinition.getType();
		log.tracef(
				"%s Starting collection element graph : %s",
				StringHelper.repeat( ">>", fetchSourceStack.size() ),
				elementDefinition.getCollectionDefinition().getCollectionPersister().getRole()
		);

		final CollectionReference collectionReference = currentCollection();
		final CollectionFetchableElement elementGraph = collectionReference.getElementGraph();

		if ( elementType.isAssociationType() || elementType.isComponentType() ) {
			if ( elementGraph == null ) {
				throw new IllegalStateException(
						"CollectionReference did not return an expected element graph : " +
								elementDefinition.getCollectionDefinition().getCollectionPersister().getRole()
				);
			}
			if ( !elementType.isAnyType() ) {
				pushToStack( (ExpandingFetchSource) elementGraph );
			}
		}
		else {
			if ( elementGraph != null ) {
				throw new IllegalStateException(
						"CollectionReference returned an unexpected element graph : " +
								elementDefinition.getCollectionDefinition().getCollectionPersister().getRole()
				);
			}
		}
	}
```

## RF_correct__VLM_invalid
### full_rq0_seed42__rq0_0145__rq0_0153
- snippets: `rq0_0145` vs `rq0_0153`
- difficulty: `hard`
- human z: -1.5143 vs -1.3132; gold: `rq0_0153`
- RF: score_a=0.3039, score_b=0.3041, margin=-0.0002, pred=`rq0_0153`, correct=True
- VLM: AB=`A`, BA=`A`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0145.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0153.png`

Code A excerpt:
```java
	
	public long
	getInterval();
	
	public long
	getMinInterval();
	
	public int
	getTimeUntilNextUpdate();

```
Code B excerpt:
```java
	 */
	public void setGadgetKey(String gadgetKey);

	/**
	 * Returns the service name of this o auth token.
	 *
	 * @return the service name of this o auth token
	 */
	@AutoEscape
	public String getServiceName();

```

## RF_wrong__VLM_correct
### full_rq0_seed42__rq0_0021__rq0_0208
- snippets: `rq0_0021` vs `rq0_0208`
- difficulty: `hard`
- human z: 1.0323 vs 1.4462; gold: `rq0_0208`
- RF: score_a=-0.0492, score_b=-0.0511, margin=0.0019, pred=`rq0_0021`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0208`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0021.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0208.png`

Code A excerpt:
```java
	protected void printFailures(Result result) {
		if (result.getFailureCount() == 0)
			return;
		if (result.getFailureCount() == 1)
			getWriter().println("There was " + result.getFailureCount() + " failure:");
		else
			getWriter().println("There were " + result.getFailureCount() + " failures:");

```
Code B excerpt:
```java
public String extractConstraintName(SQLException sqle) {
			try {
				final int sqlState = Integer.valueOf( JdbcExceptionHelper.extractSqlState( sqle ) );
				switch (sqlState) {
					// CHECK VIOLATION
					case 23514: return extractUsingTemplate( "violates check constraint \"","\"", sqle.getMessage() );
					// UNIQUE VIOLATION
					case 23505: return extractUsingTemplate( "violates unique constraint \"","\"", sqle.getMessage() );
					// FOREIGN KEY VIOLATION
					case 23503: return extractUsingTemplate( "violates foreign key constraint \"","\"", sqle.getMessage() );
					// NOT NULL VIOLATION
					case 23502: return extractUsingTemplate( "null value in column \"","\" violates not-null constraint", sqle.getMessage() );
					// TODO: RESTRICT VIOLATION
					case 23001: return null;
					// ALL OTHER
					default: return null;
				}
			}
			catch (NumberFormatException nfe) {
				return null;
			}
		}
```

### full_rq0_seed42__rq0_0069__rq0_0133
- snippets: `rq0_0069` vs `rq0_0133`
- difficulty: `hard`
- human z: 0.6504 vs 0.4328; gold: `rq0_0069`
- RF: score_a=0.4297, score_b=0.4318, margin=-0.0021, pred=`rq0_0133`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0069`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0069.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0133.png`

Code A excerpt:
```java
	/**
		Translate bsh.Modifiers into ASM modifier bitflags.
	*/
	static int getASMModifiers( Modifiers modifiers ) 
	{
		int mods = 0;
		if ( modifiers == null )
			return mods;

		if ( modifiers.hasModifier("public") )
			mods += ACC_PUBLIC;

```
Code B excerpt:
```java

		Date createDate = getCreateDate();

		if (createDate != null) {
			passwordPolicyCacheModel.createDate = createDate.getTime();
		}
		else {
			passwordPolicyCacheModel.createDate = Long.MIN_VALUE;
		}

		Date modifiedDate = getModifiedDate();

		if (modifiedDate != null) {
			passwordPolicyCacheModel.modifiedDate = modifiedDate.getTime();
		}
		else {
			passwordPolicyCacheModel.modifiedDate = Long.MIN_VALUE;
		}

		passwordPolicyCacheModel.defaultPolicy = getDefaultPolicy();

		passwordPolicyCacheModel.name = getName();

		String name = passwordPolicyCacheModel.name;

		if ((name != null) && (name.length() == 0)) {
			passwordPolicyCacheModel.name = null;
		}

		passwordPolicyCacheModel.description = getDescription();

```

### full_rq0_seed42__rq0_0096__rq0_0207
- snippets: `rq0_0096` vs `rq0_0207`
- difficulty: `medium`
- human z: -0.4425 vs 0.5387; gold: `rq0_0207`
- RF: score_a=-0.1013, score_b=-0.1080, margin=0.0067, pred=`rq0_0096`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0207`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0096.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0207.png`

Code A excerpt:
```java
			Description description= Description.createSuiteDescription(name);
			int n= ts.testCount();
			for (int i= 0; i < n; i++)
				description.addChild(makeDescription(ts.testAt(i)));

```
Code B excerpt:
```java
/**
	 * Constructs a SybaseASE157Dialect
	 */
	public SybaseASE157Dialect() {
		super();

		registerFunction( "create_locator", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "create_locator(?1, ?2)" ) );
		registerFunction( "locator_literal", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "locator_literal(?1, ?2)" ) );
		registerFunction( "locator_valid", new SQLFunctionTemplate( StandardBasicTypes.BOOLEAN, "locator_valid(?1)" ) );
		registerFunction( "return_lob", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "return_lob(?1, ?2)" ) );
		registerFunction( "setdata", new SQLFunctionTemplate( StandardBasicTypes.BOOLEAN, "setdata(?1, ?2, ?3)" ) );
		registerFunction( "charindex", new SQLFunctionTemplate( StandardBasicTypes.INTEGER, "charindex(?1, ?2, ?3)" ) );
	}
```

### full_rq0_seed42__rq0_0133__rq0_0297
- snippets: `rq0_0133` vs `rq0_0297`
- difficulty: `hard`
- human z: 0.4328 vs 0.9017; gold: `rq0_0297`
- RF: score_a=0.4318, score_b=0.4238, margin=0.0080, pred=`rq0_0133`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0297`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0133.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0297.png`

Code A excerpt:
```java

		Date createDate = getCreateDate();

		if (createDate != null) {
			passwordPolicyCacheModel.createDate = createDate.getTime();
		}
		else {
			passwordPolicyCacheModel.createDate = Long.MIN_VALUE;
		}

		Date modifiedDate = getModifiedDate();

		if (modifiedDate != null) {
			passwordPolicyCacheModel.modifiedDate = modifiedDate.getTime();
		}
		else {
			passwordPolicyCacheModel.modifiedDate = Long.MIN_VALUE;
		}

		passwordPolicyCacheModel.defaultPolicy = getDefaultPolicy();

		passwordPolicyCacheModel.name = getName();

		String name = passwordPolicyCacheModel.name;

		if ((name != null) && (name.length() == 0)) {
			passwordPolicyCacheModel.name = null;
		}

		passwordPolicyCacheModel.description = getDescription();

```
Code B excerpt:
```java
@Override
	public boolean equals(Object obj) {
		if ( this == obj ) {
			return true;
		}
		if ( !super.equals( obj ) ) {
			return false;
		}
		if ( getClass() != obj.getClass() ) {
			return false;
		}
		VersionsJoinTableRangeTestAlternateEntity other = (VersionsJoinTableRangeTestAlternateEntity) obj;
		if ( alternateValue == null ) {
			if ( other.alternateValue != null ) {
				return false;
			}
		}
		else if ( !alternateValue.equals( other.alternateValue ) ) {
			return false;
		}
		return true;
	}
```

### full_rq0_seed42__rq0_0049__rq0_0108
- snippets: `rq0_0049` vs `rq0_0108`
- difficulty: `medium`
- human z: 1.0060 vs 0.2669; gold: `rq0_0049`
- RF: score_a=0.4889, score_b=0.4977, margin=-0.0088, pred=`rq0_0108`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0049`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0049.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0108.png`

Code A excerpt:
```java
	private static String getBaseName( String className ) 
	{
		int i = className.indexOf("$");
		if ( i == -1 )
			return className;

		return className.substring(i+1);

```
Code B excerpt:
```java
		    	
		    	String	temp = "";
		    	
		    	for (int i=0;i<library_path.length();i++){
		    		
		    		char	c = library_path.charAt(i);
		    		
		    		if ( c != '"' ){
		    			
		    			temp += c;
		    			
		    		}else{
		    			
		    			changed	= true;
		    		}
		    	}
		    	
		    	library_path	= temp;
		    	
		    		// remove trailing separator chars if they exist as they stuff up
		    		// the following "
		    	
		    	while( library_path.endsWith(File.separator)){
		    	
		    		changed = true;
		    		
		    		library_path = library_path.substring( 0, library_path.length()-1 );
		    	}
		    	
		    	if ( changed ){

```

### full_rq0_seed42__rq0_0135__rq0_0299
- snippets: `rq0_0135` vs `rq0_0299`
- difficulty: `easy`
- human z: -1.2824 vs 0.3572; gold: `rq0_0299`
- RF: score_a=-0.4092, score_b=-0.4222, margin=0.0130, pred=`rq0_0135`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0299`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0135.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0299.png`

Code A excerpt:
```java
		throws com.liferay.portal.kernel.exception.PortalException,
			com.liferay.portal.kernel.exception.SystemException;

	public void deleteEntry(long entryId)
		throws com.liferay.portal.kernel.exception.PortalException,
			com.liferay.portal.kernel.exception.SystemException;

	@Transactional(propagation = Propagation.SUPPORTS, readOnly = true)
	public java.util.List<com.liferay.portlet.bookmarks.model.BookmarksEntry> getEntries(
		long groupId, long folderId, int start, int end)
		throws com.liferay.portal.kernel.exception.SystemException;

	@Transactional(propagation = Propagation.SUPPORTS, readOnly = true)
	public java.util.List<com.liferay.portlet.bookmarks.model.BookmarksEntry> getEntries(
		long groupId, long folderId, int start, int end,
		com.liferay.portal.kernel.util.OrderByComparator orderByComparator)
		throws com.liferay.portal.kernel.exception.SystemException;

	@Transactional(propagation = Propagation.SUPPORTS, readOnly = true)
	public int getEntriesCount(long groupId, long folderId)
		throws com.liferay.portal.kernel.exception.SystemException;

	@Transactional(propagation = Propagation.SUPPORTS, readOnly = true)
	public com.liferay.portlet.bookmarks.model.BookmarksEntry getEntry(
		long entryId)
		throws com.liferay.portal.kernel.exception.PortalException,
			com.liferay.portal.kernel.exception.SystemException;

	@Transactional(propagation = Propagation.SUPPORTS, readOnly = true)
	public int getFoldersEntriesCount(long groupId,
		java.util.List<java.lang.Long> folderIds)
		throws com.liferay.portal.kernel.exception.SystemException;

	@Transactional(propagation = Propagation.SUPPORTS, readOnly = true)
	public java.util.List<com.liferay.portlet.bookmarks.model.BookmarksEntry> getGroupEntries(
		long groupId, int start, int end)
		throws com.liferay.portal.k
```
Code B excerpt:
```java
private void addTransactionFactories(StrategySelectorImpl strategySelector) {
		strategySelector.registerStrategyImplementor( TransactionFactory.class, JdbcTransactionFactory.SHORT_NAME, JdbcTransactionFactory.class );
		strategySelector.registerStrategyImplementor( TransactionFactory.class, "org.hibernate.transaction.JDBCTransactionFactory", JdbcTransactionFactory.class );

		strategySelector.registerStrategyImplementor( TransactionFactory.class, JtaTransactionFactory.SHORT_NAME, JtaTransactionFactory.class );
		strategySelector.registerStrategyImplementor( TransactionFactory.class, "org.hibernate.transaction.JTATransactionFactory", JtaTransactionFactory.class );

		strategySelector.registerStrategyImplementor( TransactionFactory.class, CMTTransactionFactory.SHORT_NAME, CMTTransactionFactory.class );
		strategySelector.registerStrategyImplementor( TransactionFactory.class, "org.hibernate.transaction.CMTTransactionFactory", CMTTransactionFactory.class );
	}
```

### full_rq0_seed42__rq0_0065__rq0_0156
- snippets: `rq0_0065` vs `rq0_0156`
- difficulty: `medium`
- human z: 1.1640 vs 0.2482; gold: `rq0_0065`
- RF: score_a=0.3887, score_b=0.4038, margin=-0.0151, pred=`rq0_0156`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0065`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0065.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0156.png`

Code A excerpt:
```java
  /**
   * Add one zero if neccessary
   * @param number
   * @return
   */
  private CharSequence addZero(int number) {
    StringBuilder builder = new StringBuilder();
    
    if (number < 10) {
      builder.append('0');
    }
    
    builder.append(Integer.toString(number));

```
Code B excerpt:
```java

		for (KaleoTimer model : models) {
			soapModels.add(toSoapModel(model));
		}

		return soapModels.toArray(new KaleoTimerSoap[soapModels.size()]);
	}

	public KaleoTimerSoap() {
	}

```

### full_rq0_seed42__rq0_0074__rq0_0205
- snippets: `rq0_0074` vs `rq0_0205`
- difficulty: `hard`
- human z: 0.3212 vs 0.5387; gold: `rq0_0205`
- RF: score_a=0.4444, score_b=0.4290, margin=0.0154, pred=`rq0_0074`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0205`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0074.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0205.png`

Code A excerpt:
```java
      
      out.writeObject(device.getDriver().getClass().getName());
      out.writeObject(device.getName());
      
      device.writeData(out);

```
Code B excerpt:
```java
/**
	 * Constructs a Oracle8iDialect
	 */
	public Oracle8iDialect() {
		super();
		registerCharacterTypeMappings();
		registerNumericTypeMappings();
		registerDateTimeTypeMappings();
		registerLargeObjectTypeMappings();
		registerReverseHibernateTypeMappings();
		registerFunctions();
		registerDefaultProperties();
	}
```

### full_rq0_seed42__rq0_0180__rq0_0181
- snippets: `rq0_0180` vs `rq0_0181`
- difficulty: `hard`
- human z: -0.7412 vs -0.3984; gold: `rq0_0181`
- RF: score_a=0.1234, score_b=0.1031, margin=0.0203, pred=`rq0_0180`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0181`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0180.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0181.png`

Code A excerpt:
```java

		String emailMessageAddedSubjectPrefix = getParameter(
			actionRequest, "emailMessageAddedSubjectPrefix");
		String emailMessageAddedBody = getParameter(
			actionRequest, "emailMessageAddedBody");

		if (Validator.isNull(emailMessageAddedSubjectPrefix)) {
			SessionErrors.add(actionRequest, "emailMessageAddedSubjectPrefix");
		}
		else if (Validator.isNull(emailMessageAddedBody)) {
			SessionErrors.add(actionRequest, "emailMessageAddedBody");
		}
	}

	protected void validateEmailMessageUpdated(ActionRequest actionRequest)
		throws Exception {

		String emailMessageUpdatedSubjectPrefix = getParameter(
			actionRequest, "emailMessageUpdatedSubjectPrefix");
		String emailMessageUpdatedBody = getParameter(
			actionRequest, "emailMessageUpdatedBody");

		if (Validator.isNull(emailMessageUpdatedSubjectPrefix)) {
			SessionErrors.add(
				actionRequest, "emailMessageUpdatedSubjectPrefix");
		}
		else if (Validator.isNull(emailMessageUpdatedBody)) {
			SessionErrors.add(actionRequest, "emailMessageUpdatedBody");
		}
	}

```
Code B excerpt:
```java
/**
 * @author Brian Wing Shun Chan
 */
public class UnitConverterTestPlan extends BaseTestSuite {

	public static Test suite() {
		TestSuite testSuite = new TestSuite();

		testSuite.addTest(PortletTestPlan.suite());
		testSuite.addTest(UnitTestPlan.suite());

```

### full_rq0_seed42__rq0_0027__rq0_0250
- snippets: `rq0_0027` vs `rq0_0250`
- difficulty: `hard`
- human z: 1.0455 vs 1.2647; gold: `rq0_0250`
- RF: score_a=0.0845, score_b=0.0636, margin=0.0209, pred=`rq0_0027`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0250`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0027.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0250.png`

Code A excerpt:
```java
        boolean    hasReturnValue;

        outlen = parameters.length;
        offset = 0;

```
Code B excerpt:
```java
@Override
	protected void prepareTest() throws Exception {
	    Session s = openSession();
	    Transaction t = s.beginTransaction();
	    Child child_1_1 = new Child( "achild1-1");
	    Child child_1_2 = new Child( "ychild1-2");
	    Child child_1_3 = new Child( "dchild1-3");
	    Child child_2_1 = new Child( "bchild2-1");
	    Child child_2_2 = new Child( "cchild2-2");
	    Child child_2_3 = new Child( "zchild2-3");
	
	    s.save( child_1_1 );
	    s.save( child_2_1 );
	    s.save( child_1_2 );
	    s.save( child_2_2 );
	    s.save( child_1_3 );
	    s.save( child_2_3 );
	
	    s.flush();
	
	    Parent p1 = new Parent( "parent1" );
	    p1.addChild( child_1_1 );
	    p1.addChild( child_1_2 );
	    p1.addChild( child_1_3 );
	    s.save( p1 );
	
	    Parent p2 = new Parent( "parent2" );
	    p2.addChild( child_2_1 );
	    p2.addChild( child_2_2 );
	    p2.addChild( child_2_3 );
	    s.save( p2 );
	
	    t.commit();
	    s.close();
	}
```

## RF_wrong__VLM_wrong
### full_rq0_seed42__rq0_0075__rq0_0257
- snippets: `rq0_0075` vs `rq0_0257`
- difficulty: `easy`
- human z: -0.8903 vs 0.3572; gold: `rq0_0257`
- RF: score_a=-0.7107, score_b=-0.7125, margin=0.0018, pred=`rq0_0075`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0075`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0075.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0257.png`

Code A excerpt:
```java
        
        File data = new File(Plugin.getPluginManager().getTvBrowserSettings().getTvBrowserUserHome()  + File.separator + 
                "CaptureDevices" + File.separator + mCount + ".dat");
        
        ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(data));
        
        dev.writeData(stream);

```
Code B excerpt:
```java
private void initOrdinaryPropertyPaths(Mapping mapping) throws MappingException {
		for ( int i = 0; i < getSubclassPropertyNameClosure().length; i++ ) {
			propertyMapping.initPropertyPaths( getSubclassPropertyNameClosure()[i],
					getSubclassPropertyTypeClosure()[i],
					getSubclassPropertyColumnNameClosure()[i],
					getSubclassPropertyColumnReaderClosure()[i],
					getSubclassPropertyColumnReaderTemplateClosure()[i],
					getSubclassPropertyFormulaTemplateClosure()[i],
					mapping );
		}
	}
```

## RF_wrong__VLM_invalid
### full_rq0_seed42__rq0_0021__rq0_0308
- snippets: `rq0_0021` vs `rq0_0308`
- difficulty: `easy`
- human z: 1.0323 vs -0.9134; gold: `rq0_0021`
- RF: score_a=-0.0492, score_b=-0.0488, margin=-0.0003, pred=`rq0_0308`, correct=False
- VLM: AB=`B`, BA=`B`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0021.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0308.png`

Code A excerpt:
```java
	protected void printFailures(Result result) {
		if (result.getFailureCount() == 0)
			return;
		if (result.getFailureCount() == 1)
			getWriter().println("There was " + result.getFailureCount() + " failure:");
		else
			getWriter().println("There were " + result.getFailureCount() + " failures:");

```
Code B excerpt:
```java
public static <T> JaxbRoot<T> unmarshallXml(String fileName, String schemaName, Class<T> clazz, ClassLoaderService classLoaderService)
            throws JAXBException {
        Schema schema = getMappingSchema( schemaName, classLoaderService );
        InputStream in = classLoaderService.locateResourceStream( fileName );
        JAXBContext jc = JAXBContext.newInstance( clazz );
        Unmarshaller unmarshaller = jc.createUnmarshaller();
        unmarshaller.setSchema( schema );
        StreamSource stream = new StreamSource( in );
        JAXBElement<T> elem = unmarshaller.unmarshal( stream, clazz );
        Origin origin = new Origin( null, fileName );
        return new JaxbRoot<T>( elem.getValue(), origin );
    }
```

